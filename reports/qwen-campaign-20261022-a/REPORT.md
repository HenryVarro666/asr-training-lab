# Qwen ASR campaign

Status: complete; stage: evaluations_complete.

Each benchmark starts from the same pinned base model. Training-update timing excludes final generation evaluation and checkpoint writes.

| Experiment | GPUs | Status | Audio sec / wall sec | Peak project GiB | Min observed free GiB |
|---|---:|---|---:|---:|---:|
| E1-single-b2 | 1 | measured | 92.40828676509733 | 35.637484550476074 | 54.5576171875 |
| E2-dual-b2 | 2 | measured | 177.82783941372216 | 43.21487903594971 | 24.873046875 |
| E3-dual-b4 | 2 | measured | 257.2953106723787 | 44.511733531951904 | 24.0625 |
| E4-dual-b8 | 2 | measured | 290.572487442234 | 47.2247896194458 | 19.88671875 |

Selected dual-GPU configuration: qwen_batch8.yaml.

A second learning-rate run is not launched. Its recommendation remains pending analysis of the first completed training and evaluation matrix.

| Evaluation | Baseline WER | Final WER | Final minus baseline (percentage points) |
|---|---:|---:|---:|
| fixed128 | 1.5835% | 2.3064% | +0.7229 |
| dev_clean | 1.6122% | 2.3729% | +0.7607 |
| dev_other | 3.0850% | 4.9850% | +1.9000 |
