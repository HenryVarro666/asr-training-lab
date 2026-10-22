# 双 H100 ASR 首轮实验结果

更新：2026-10-22T17:45:23.520665+00:00。**首轮训练和评测已完成，但识别质量退化，当前微调模型不应替换预训练基线。** 后续代理调用触及用量限制，目标状态为 `usageLimited`；没有启动第二轮训练。

## 完成的工作

- Astra 6 完成方案与训练流程审阅；GPT-6.1 Sol 完成训练、数据处理和实验编排代码。
- 代码固定为 `cf30f45e0beb005f43520c33e2ddad73ad67a01e`；远端实际运行前34项测试及5个subtests通过。
- 数据/模型/环境/日志/checkpoints全部位于 `/mnt/raid/asr-lab`。准备LibriSpeech train-clean-100共28,539条、100.591小时，及独立dev-clean/dev-other。
- 主模型 `Qwen/Qwen3-ASR-1.7B`，revision `7278e1e70fe206f11671096ffdd38061171dd6e5`；加载后唯一参数实测2,038,052,480。此前HF元数据估算2.349B不能作为实际训练参数数目。
- 主训练双卡DDP、BF16 autocast、FP32权重/梯度/Adam状态、SDPA、activation checkpointing；microbatch8 × 2卡 × accumulation2，总batch32；lr1e-5，一轮。
- 主训练完成892次optimizer更新、epoch=1.0，用时25.65分钟。含4个效率实验和完整评测的campaign总时长53.87分钟。
- 原有GPU0 PersonaPlex进程2154793仍在，未停止。最终检查只见其GPU占用，训练工作已退出；RAID仍约3.1TiB可用。

## 两张H100的实测利用

以下使用相同预训练初始化、固定2048条子集、总batch32、40步训练，其中11–40步计时。吞吐仅含训练更新，不含模型加载/保存和生成评测。

| 配置 | GPU数 | 音频秒/墙钟秒 | 项目峰值显存GiB/卡 | 最小空闲显存GiB |
|---|---:|---:|---:|---:|
| E1-single-b2 | 1 | 92.41 | 35.64 | 54.56 |
| E2-dual-b2 | 2 | 177.83 | 43.21 | 24.87 |
| E3-dual-b4 | 2 | 257.30 | 44.51 | 24.06 |
| E4-dual-b8 | 2 | 290.57 | 47.22 | 19.89 |

双卡batch2对单卡相同microbatch约1.92倍吞吐；双卡batch8约为单卡batch2的3.14倍，后者同时包含microbatch效率改善，不能称作纯双卡线性扩展效率。最终采用batch8；最少仍有约19.9GiB空闲显存。型号虽为H100 NVL，实测拓扑显示NODE，未假设有NVLink。

## 生成式识别结果

同一manifest、样本ID、参考词分母、Whisper EnglishTextNormalizer版本、greedy解码和batch16；无生成截断。数据均为合格短音频，**不是包含所有超长录音的完整官方benchmark**。

| 评测集 | 原始模型WER | 微调后WER | 变化（百分点，正数更差） | 样本数 | 参考词数 |
|---|---:|---:|---:|---:|---:|
| fixed128 | 1.5835% | 2.3064% | +0.7229 | 128 | 2905 |
| dev_clean | 1.6122% | 2.3729% | +0.7607 | 2694 | 54027 |
| dev_other | 3.0850% | 4.9850% | +1.9000 | 2857 | 51053 |

微调后dev-clean从871个词错误增加至1282个，dev-other从1575增加至2545个。主要增加的是替换和插入错误。训练平均loss约0.0658不能代替识别质量结论。这里仅报告描述性差异，尚未完成paired bootstrap或误差分类，因此不声明统计显著性或确定的退化原因。

独立Whisper-small参考：同128条WER3.3391%；合格dev-clean2694条WER3.4057%。这是不同模型的参考，不改变Qwen自身微调退化的结论。test-clean/test-other未用于调参或本轮结果。

## 可恢复产物和监控证据

最终checkpoint：`/mnt/raid/asr-lab/runs/qwen-campaign-20261022-a/main-epoch1/checkpoint-892`。已检查权重、optimizer、scheduler、trainer_state、两rank RNG和checkpoint metadata都存在。另有真实GPU step2→4恢复验证，707个存储tensor全部有限，最大绝对差8.90e-6；不声称CUDA逐位一致。

- [完整campaign报告](reports/qwen-campaign-20261022-a/REPORT.md)
- [机器可读执行状态](reports/qwen-campaign-20261022-a/campaign.json)
- [完整主训练状态](reports/qwen-campaign-20261022-a/main-epoch1/status.json)
- [恢复验证](reports/validation/resume-verification.json)
- [计划书](ASR_TRAINING_PLAN.md)
- [固定环境](reports/environment/remote-pip-freeze.txt)

原始逐句预测、指标、配置、训练曲线和GPU/磁盘采样已同步到本地 `reports/qwen-campaign-20261022-a/`。大权重留在服务器，不提交Git。TensorBoard文件保留，SSH转发存活时可通过 http://127.0.0.1:16006/ 查看。

campaign已结束且用量限制阻止后续代理实验，15分钟自动化 `h100-asr` 已删除，避免无意义轮询。监控阶段未重启训练、扩大预算或停止其他服务。

## 未完成的后续工作

先对生成错误和可用早期checkpoint做诊断，再决定降低lr、缩短训练或受控LoRA对照；不继续增加epochs。`docs/ADAPTIVE_EXPERIMENT_POLICY.md`、`scripts/analyze_asr_predictions.py` 及其测试是代理留下的待验收草稿，不能当作已运行的CI或预注册分析。第二轮训练、中文/混合语料、流式评测和独立test评测均未执行。
