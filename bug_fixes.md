# Bug Fixes

## 1. `config.py` — `set_output_prefix` used two `if` statements instead of `if/elif`
When both `--patch_model` and `--eval_model` were true, the second `if` always overwrote the first, causing the output prefix to be the long eval path (`atp/_eval/_steer`) even during the patching phase. Patching results were saved to the wrong directory. Fixed by changing the second `if` to `elif` so patching uses `atp/` and eval uses `atp/_eval/_steer`.

## 2. `eval/logits_handler.py` — load path stripped too many directory components
`load_logits` computed the load path by calling `[:-3]` on the eval prefix, intending to strip `_steer/_eval/atp` and land at the base results dir. But since bug #1 caused files to be saved under the full eval prefix, this stripped the wrong levels. Fixed by using `config.get_output_prefix()` directly (no stripping), consistent with where files are now saved.

## 3. `config.py` — `--eval_test false` parsed as truthy string
`--eval_test` uses `nargs="?"`, so passing `--eval_test false` gives argparse the string `"false"` rather than the boolean `False`. Downstream `isinstance(args.eval_test, bool)` checks then fail. Fixed by adding post-parse conversion: strings `"false"`/`"true"` are converted to their boolean equivalents immediately after `parse_args()`.

## 4. `data_handler.py` — `steering_qs_toks` only set under `eval_test` branch
`steering_qs_toks` was assigned inside `elif self.config.args.eval_test:`, so it was never set when `eval_test=False`. `steering_reps_cache()` accesses it unconditionally, causing an `AttributeError`. Fixed by moving the `steering_qs_toks` assignment to a standalone `if self.config.args.eval_model:` block after the `patch_model`/`eval_model` branching, so it is always set when eval is running.

## 5. `gen_data.py` — hardcoded `-long` suffixes in output file paths
When `source=lie-long` was passed in, the file path logic appended an extra `-long` to both the directory and filename (e.g. `lie-long-long/lie-long-long-desired-all.jsonl`). Fixed by removing the hardcoded `-long` additions and using `source` and `gen` variable values directly, which already contain `-long` when passed in.

## 6. `run.sh` — `--eval_test` not explicitly passed when false
When `EVAL_TEST=false`, the flag was simply omitted, relying on the argparse default. This caused ambiguity with the `nargs="?"` setup. Fixed by explicitly passing `--eval_test false` when `EVAL_TEST=false` (works in conjunction with fix #3).
