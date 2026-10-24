# 本轮交付验收

最新准备阶段：[逐token概率诊断](token-probability/README.md)已保存合同、代码、固定轨迹/官方来源、CPU测试、独立最终审阅、部署SHA和资源快照。GPU1占用导致`ready_blocked_gpu_busy`；0模型调用/0更新/0真实概率行/0结果图，无自动队列。现有实验结果和历史归档保持原记录。

本页按当前goal和主计划第9、11节核对实际交付；不把待执行计划、代码或GPU显存占用作为完成证据。用户已取消耗尽token/usage，后续采用单阶段有界实验。2026-10-23归档：零更新控制已通过；新增Early100训练完成，但完整clean截断导致评测门槛失败，other未执行。以下将完成与未验收部分分开；研究质量结论仍为微调退化。

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
| Early100双卡轨迹与检查点 | `early100/README.md`、`run/training/trajectory-validation.json`、三个checkpoint完成标记 | 100更新/892原调度/45warmup通过；25/50模型导出、100完整恢复状态已验；未另做恢复训练 |
| Early100生成与失败门槛 | `early100/run/campaign.json`、三份fixed128配对统计、`truncation-audit.json` | 三个固定128通过且早期退化；clean1截断，campaign failed，other未执行，不计为完整阶段通过 |
| 固定512-token截断探针 | `truncation-probe/README.md`、`run/`、`execution-summary.json`、独立CPU审计 | 3次生成／33条消费／0更新／130.69秒；3次目标raw一致且无EOS，生成重复至512后由parser压短；旧failed不变，未确定训练根因 |
| 四检查点重复出现对照 | `repetition-onset/README.md`、`run/`、`independent-audit.json` | 4次调用／0更新／99.16秒；base/25/50正常、100首次已测强循环及512无EOS；非精确首发更新或训练根因；100本地/14远端CPU tests通过 |
| 独立可复算图片 | `figures/figure_manifest.json`、`figures/early100/figure_manifest.json` | 原三组+新增两组，每组PNG/SVG/PDF；输入与导出SHA、重复渲染、视觉检查；没有虚构full-dev图 |
| 监控与原服务保护 | 各run的`monitor/`、`early100/final-host-snapshot.json` | 已保存过程采样和完成快照；GPU0 PersonaPlex保留；全部ASR进程已退出；既有告警原样保留 |
| 可查看训练曲线 | `monitoring/main-epoch1-tensorboard/`、`monitoring/tensorboard-verification.json` | 事件实际解析出892条train loss/grad/lr、8条eval loss；非仅空TensorBoard目录 |
| 可复现仓库 | README、配置、CLI、许可、CI、环境冻结、测试和上述收据 | Early100前本地72项、远端专属12项通过；本地Git交付与跨机bundle备份，未发布GitHub、未公开模型或音频 |

## 质量和范围

流程完成与研究收益分开：首轮微调后WER上升，当前仍选择原始Qwen作为开发集质量基线。此结果不能包装成“性能提升”或“已训练出更好的通用语音模型”。

中文、960h扩展、原生streaming、TTS/RL、自研projector及TP/FSDP在当前计划中是后续或有条件的扩展；未执行，也不计入已交付能力。独立test和人工听音尚未执行。若以后宣称最终模型验收、独立质量确认或流式能力，必须完成相应评测，不能用本轮dev诊断替代。

首轮和前两项控制均已交付。Early100保留训练成功、固定128早期退化、完整clean截断以及other未执行的真实边界，不能计为完整五评测验收通过。新增异常样本探针及四检查点对照均已按各自预算完成并归档，定位了生成重复及parser压缩，并将最早受测强循环定位至100步；未修复模型、确定精确首次异常更新或训练原因。不能因为GPU空闲或仍有额度，就自动追加训练和超参数扫描。
