import os
import json
import torch
import einops
import pandas as pd
import matplotlib.pyplot as plt

def load_logits(config, data_handler, which_patch, model_handler):
    prefix = config.get_output_prefix()
    heads_dir = f"{prefix}/heads"
    logits_path = f"{heads_dir}/{which_patch}"
    all_logits = None

    name = 'numerator_1'
    if os.path.exists(f"{heads_dir}/{name}_{which_patch}.pt"):
        all_logits = torch.load(f"{heads_dir}/{name}_{which_patch}.pt")
    else:
        print('Path does not exist {}, computing logits afresh.'.format(f"{heads_dir}/{name}_{which_patch}.pt"))
        for i in range(data_handler.LEN):
            try:
                with open(f"{logits_path}_{i}.pt", 'rb') as f:
                    logits = torch.load(f)
                    logits = logits.squeeze().unsqueeze(-1)
                    all_logits = logits if all_logits is None else torch.cat([all_logits, logits], dim=-1)
            except Exception as e:
                print(f"Could not load logits from {logits_path}_{i}.pt, skipping this file. ", e)
                continue

        all_logits = einops.reduce(all_logits, 'l (n m) b -> l n b', 'sum', n=model_handler.num_heads)
        plot_logit_metrics(config, model_handler, all_logits, name, which_patch)
        torch.save(all_logits, f"{heads_dir}/{name}_{which_patch}.pt")
    return all_logits

def get_top_k_layer_and_head(patches, top_k):
    if isinstance(patches, str):
        patches = torch.load(patches)
    patches = patches.to(torch.float32)
    patches = patches.mean(dim=-1)
    flat = patches.view(-1)
    top_values, top_indices = flat.topk(k=int(top_k * flat.numel()))
    layer_indices = top_indices // patches.shape[1]
    neuron_indices = top_indices % patches.shape[1]
    df = pd.DataFrame({
        'layer': layer_indices.numpy(),
        'neuron': neuron_indices.numpy(),
        'value': top_values.numpy()
    })
    return df.sort_values(by=['layer', 'neuron'])

def plot_logit_metrics(config, model_handler, metric, name, which_patch):
    metric = metric.to(torch.float32)
    metric = metric.mean(dim=-1)

    plt.imshow(metric, cmap="viridis")
    plt.colorbar(label='Indirect Effect size')
    plt.ylabel("Layers")
    plt.xlabel("Heads")
    plt.grid(True)
    plt.title(f"Post-patch logit difference: {config.args.base}")
    plt.xticks(ticks=range(model_handler.num_heads))
    plt.yticks(ticks=range(model_handler.num_layers))
    plt.tight_layout()
    os.makedirs(config.get_output_prefix(), exist_ok=True)
    plt.savefig(f"{config.get_output_prefix()}/{name}_heatmap.png")
    plt.close()
