# Frozen-trajectory token probability probe

Status: failed; stage: base.

At most3 native-observation generations and9 explicit empty-cache teacher forwards; no training, retries or additional samples.
Three fixed saved trajectories remain27/37/512 IDs; no re-tokenization. Native output_logits/output_scores are the only allowed observation differences.
Full-vocabulary FP32 continuation rows are archived; no prompt logits or full probability matrices are saved.
Same-prefix probabilities describe model behavior; they do not identify training causes.

Failure: RuntimeError: Physical GPU1 has other compute applications; no launch: [['GPU-d8788c54-b9da-d038-e5bc-aa3b718ef62c', '2801775', '/home/huaqinzhao/miniconda3/envs/loopvlm-h100/bin/python']]
