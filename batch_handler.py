import sys
sys.path.insert(1, '../atp/')

class BatchHandler:
    def __init__(self, config, data_handler, start=None, stop=None):
        self.config = config
        self.data_handler = data_handler
        self.batch_size = config.args.batch_size

        if not start or not stop:
            start = 0
            stop = self.batch_size

        self.start = start
        self.stop = stop

        if self.config.args.eval_model:
            if self.config.args.eval_test:
                self.base_qs_toks = {
                    'test': { "input_ids": self.data_handler.base_qs_toks['test']["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_qs_toks['test']["attention_mask"][self.start:self.stop]}
                }

    def update(self, start=None, stop=None):
        if start is None or stop is None:
            self.start = self.start + self.batch_size
            self.stop = self.start + self.batch_size
        else:
            self.start = start
            self.stop = stop

        if self.config.args.eval_model:
            if self.config.args.eval_test:
                self.base_qs_toks = {
                    'test': { "input_ids": self.data_handler.base_qs_toks['test']["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_qs_toks['test']["attention_mask"][self.start:self.stop]}
                }
