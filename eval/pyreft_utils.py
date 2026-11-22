import pyreft
import torch
from tqdm import trange
import torch
from collections import OrderedDict

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from transformers.activations import ACT2FN

class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class SteeringLoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None, S=1, prompt_mask=None
    ):
        # print(f"SteeringLoreftIntervention forward with S={S}")
        # if  S != 1:
        #     print(f"SteeringLoreftIntervention scaling with S={S}")
        rotated_base = self.rotate_layer(base)
        steering_vector = torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        ) # [B, 1, T, D]
        eps = 1e-12
        steering_vector  = steering_vector / (torch.norm(steering_vector , dim=-1, keepdim=True) + eps)
        if prompt_mask is not None:
            steering_vector = steering_vector * prompt_mask[:, None, :, None].to(steering_vector.dtype)
        output = base + S * steering_vector
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True # we must match!
        
        return

def get_reft_layers_config(topk_df, model):
    representations = []
    for _, row in topk_df.iterrows():
        layer = row['layer']
        representations.append({
            "layer": int(layer),
            "component": "head_attention_value_output",
            "unit": f"h.pos",
            "low_rank_dimension": 1,
            "intervention": SteeringLoreftIntervention(embed_dim=model.config.hidden_size // model.config.num_attention_heads,
                                                     low_rank_dimension=1)
        })
    return pyreft.ReftConfig(representations=representations)

def get_intervention_locations(topk_df, input_ids):
    """
    Helper method for getting intervention locations for specific attention heads + token positions

    input_ids : (batch_size, seq_len)

    returns: (num_interventions, 2, batch_size, seq_len)
    """
    tokens = list(range(input_ids.shape[-1]))
    heads = []
    for _, row in topk_df.iterrows():
        heads.append(int(row['neuron']))
    return [ # intervention layer
        [ # unit/position layer
            [ # token layer
                [head]
            ] * input_ids.shape[0], # batch layer
            [ # token layer
                tokens
            ] * input_ids.shape[0] # batch layer
        ]
    for head in heads]

def reft_train(
    topk_df, pv_model, cf_data, labels, prompt_mask,
    batch_size=8, lr=0.0001, num_epochs=10, device='cuda', display_bar=True
):
    pv_model.train()
    optimizer = torch.optim.Adam(pv_model.parameters(), lr=lr)

    tolerance = 1e-4       # Minimum loss change to be considered an improvement
    patience = 5        # How many epochs to wait for improvement before stopping
    max_epochs = 500        # Just in case convergence is never reached

    best_loss = float('inf')
    no_improve_epochs = 0
    epoch = 0

    print('Number of samples ', cf_data['input_ids'].shape[0])
    while epoch < max_epochs:
        epoch += 1
        epoch_loss = 0.0
        num_batches = 0

        with trange(0, len(cf_data['input_ids']), batch_size, desc=f'Training (Epoch {epoch})', disable=not display_bar) as progress_bar:
            print('Trange ', progress_bar)
            for batch_index in progress_bar:
                optimizer.zero_grad()

                batch = {
                    'input_ids': cf_data['input_ids'][batch_index:batch_index+batch_size],
                    'attention_mask': cf_data['attention_mask'][batch_index:batch_index+batch_size]
                }

                base_ids = batch['input_ids']
                base_attn_mask = batch['attention_mask']
                base = {
                    'input_ids': base_ids.to(device),
                    'attention_mask': base_attn_mask.to(device)
                }
                batch_labels = labels[batch_index:batch_index+batch_size]
                base_intervention_locations = get_intervention_locations(topk_df, base_ids)
                _, cf_outputs = pv_model(
                    base,
                    None,
                    {
                        'sources->base': (
                            None,  # copy from
                            base_intervention_locations  # paste to
                        )
                    },
                    labels=batch_labels.to(device),
                    intervention_additional_kwargs={'prompt_mask': prompt_mask[batch_index:batch_index+batch_size].to(device)}
                )

                loss_value = cf_outputs.loss.item()
                epoch_loss += loss_value
                num_batches += 1

                progress_bar.set_postfix({'loss': loss_value})
                cf_outputs.loss.backward()
                optimizer.step()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.6f}")

        if avg_loss < best_loss - tolerance:
            best_loss = avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Stopping early after {epoch} epochs due to convergence.")
                break
    return pv_model