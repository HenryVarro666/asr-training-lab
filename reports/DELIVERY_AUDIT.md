# 本轮交付验收

本页按当前goal和主计划第9、11节核对实际交付；不把待执行计划、代码或GPU显存占用作为完成证据。用户已取消耗尽token/usage，后续采用单阶段有界实验。2026-10-22（Chicago）零更新控制已完成并经Astra独立复核，以下当前阶段要求均有实际证据；研究质量结论仍为微调退化。

| 要求 | 当前证据 | 判定／边界 |
|---|---|---|
| Astra策略审阅、Sol代码实施 | `../ASR_TRAINING_PLAN.md`、`../docs/diagnosis_model_audit.md`、训练／数据／评测代码与测试 | 已交付；新控制实现及真实结果也经Astra审阅，无阻断项 |
| 真实数据准备与隔离 | `data/archives.json`、`data/manifest-report.json`，主run固定manifest指纹 | 已完成首轮100h与两项dev；未下载20k小时 |
| Whisper对照 | `baselines/whisper-shared128/`、`baselines/whisper-full-clean/verification.json` | 已完成预训练参考；共同128及2694条合格clean；未声称Whisper整轮微调 |
| Qwen双卡主训练 | `qwen-campaign-20261022-a/main-epoch1/status.json`、`run.json`、`metrics.jsonl` | 已完成892更新／epoch1.0；FP32参数与Adam、BF16 autocast、双卡DDP |
| 数据覆盖口径 | `data/training-coverage.json` | 28539唯一条目、28544次消费；五次额外消费与批次补齐一致，逐条曝光ID未保存 |
| 两卡效率与容量 | `qwen-campaign-20261022-a/experiment_results.json`及E1–E4原始记录 | 单卡92.41、双卡同microbatch177.83、双卡batch8达290.57音频秒/秒；不同batch收益不冒充纯并行收益 |
| 训练正确性与恢复 | `validation/resume-verification.json`、`validation/resume-run/`、训练契约测试 | 实机step2→4恢复，707个tensor有限，最大绝对差8.90e-6；不是CUDA位级等价声明 |
| 原始／结束生成评测 | `qwen-campaign-20261022-a/evaluation/` | 共同ID/参考文本/normalizer，原始与892的合格clean2694、other2857均已完成；微调WER退化 |
| 退化诊断与现存checkpoint比较 | `diagnosis/README.md`、`diagnosis/checkpoint800/`及配对统计 | 已完成5551条诊断与800评测；退化800时已存在，未确定机制 |
| 新增零更新保存／重载控制 | `zero-update-control/README.md`、`verification.json`、`comparison.json` | 真实通过；136.79秒、0更新，128条raw token及状态／输入／有效解码一致；WER同为1.5835% |
| 监控与原服务保护 | 各run的`monitor/`、`zero-update-control/final-host-snapshot.json` | 已保存过程采样和完成快照；GPU0 PersonaPlex保留；全部ASR进程已退出；既有告警原样保留 |
| 可查看训练曲线 | `monitoring/main-epoch1-tensorboard/`、`monitoring/tensorboard-verification.json` | 事件实际解析出892条train loss/grad/lr、8条eval loss；非仅空TensorBoard目录 |
| 可复现仓库 | README、配置、CLI、许可、CI、环境冻结、测试和上述收据 | 本地60项、远端控制11项测试通过；本地Git交付，未发布GitHub、未公开模型或音频 |

## 质量和范围

流程完成与研究收益分开：首轮微调后WER上升，当前仍选择原始Qwen作为开发集质量基线。此结果不能包装成“性能提升”或“已训练出更好的通用语音模型”。

中文、960h扩展、原生streaming、TTS/RL、自研projector及TP/FSDP在当前计划中是后续或有条件的扩展；未执行，也不计入已交付能力。独立test和人工听音尚未执行。若以后宣称最终模型验收、独立质量确认或流式能力，必须完成相应评测，不能用本轮dev诊断替代。

当前有限阶段已交付：两卡真实训练、效率与恢复验证、基线／生成评测、监控、诊断和控制，以及可复现记录均已完成；结果为有效的负向微调实验。下一项候选是保持原调度的早期轨迹验证，需独立实验单，尚未启动。不能因为GPU空闲或仍有额度，就自动追加训练和超参数扫描。
