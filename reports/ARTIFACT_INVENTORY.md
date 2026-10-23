# ASR实验产物归档索引

本地工作目录：`/Users/caochao/Documents/ChatGPT/train`；远端资源根：`/mnt/raid/asr-lab`。代码、文档、实际收据、原始逐句预测、监控记录与图片分别归档，结构合理。此前缺少独立可导出的图片，本轮补齐原实验三组与Early100两组PNG/SVG/PDF及逐图来源指纹。随后已执行获授权的Early100短程双卡训练；没有下载新模型/音频或公开推送。

归档日期2026-10-23。Early100训练完成，完整clean有1条截断触发验收失败，other未执行；状态与证据见[本阶段报告](early100/README.md)。原三组图继续保持不变，新目录只展示已通过固定128结果和实际100步loss/LR。

| 种类 | 本地位置 | 保存内容与边界 |
|---|---|---|
| 代码、配置、测试 | `../src/asr_lab/`、`../scripts/`、`../configs/`、`../tests/` | Qwen与Whisper数据/训练/恢复/评测、有限runner、CPU契约测试；源码不等于GPU验证 |
| 计划与阶段进度 | `../README.md`、`../ASR_TRAINING_PLAN.md`、`../EXPERIMENT_STATUS.md`、`../docs/NEXT_STEPS.md` | 入口、阶段授权、完成状态和下一步；单阶段继续，不恢复无限自动队列 |
| 环境与数据来源 | `environment/remote-pip-freeze.txt`、`data/archives.json`、`data/manifest-report.json` | 实际依赖版本、官方归档MD5、本地SHA256、解码PCM/隔离规则、合格与拒绝数量；不是新下载收据 |
| 实际数据消费 | `data/training-coverage.json` | 28,539唯一训练条目与28,544次消费，额外5次与batch padding一致；未记录逐条曝光ID，不能声称每条恰好一次 |
| Whisper参考 | `baselines/whisper-shared128/`、`baselines/whisper-full-clean/` | 同128与2,694条合格clean的原始/规范化预测、S/D/I和核验；不是Whisper整轮微调 |
| 主实验进度与恢复点说明 | `qwen-campaign-20261022-a/campaign.json`、`main-epoch1/status.json`、`main-epoch1/run.json`、`main-epoch1/trainer_state.json` | 892更新、epoch1.0、双卡DDP、配置/源码/模型/manifest身份及checkpoint路径 |
| 曲线与资源原始日志 | `qwen-campaign-20261022-a/main-epoch1/metrics.jsonl`、各run的`monitor/` | loss/LR/grad/实际吞吐、GPU UUID/显存/温度/功率、磁盘和告警；末次采样不是终态，stage/campaign receipt才是结束依据 |
| TensorBoard事件 | `monitoring/main-epoch1-tensorboard/`、`monitoring/tensorboard-verification.json` | 已同步实际events文件；解析892条train loss/LR/grad记录、8条固定128 dev loss；既有事件没有image/audio摘要，不应把目录存在当成保存了图 |
| 单/双卡效率 | `qwen-campaign-20261022-a/E1-single-b2/`至`E4-dual-b8/`、`experiment_results.json` | 实际40步测量、丢弃10步warmup、30步窗口、global32、峰值及空闲显存；3.14×包含microbatch变化，纯相同microbatch对照约1.92× |
| 原始/结束生成评测 | `qwen-campaign-20261022-a/evaluation/` | 相同manifest、ID、参考词分母、normalizer与解码条件的原始/892预测和指标；本轮结果是微调退化 |
| 恢复实测 | `validation/resume-verification.json`、`validation/resume-run/` | 真正step2→4恢复及707个tensor比较；最大差8.90e-6，不是CUDA位级一致声明 |
| 文本与800步诊断 | `diagnosis/README.md`、`diagnosis/*.paired.json`、`diagnosis/*.jsonl`、`diagnosis/checkpoint800/` | 全样本复算、固定审阅清单、speaker-bootstrap、现存800步完整合格dev；不是人工听音或独立test |
| 零更新保存/重载控制B | `zero-update-control/README.md`、`verification.json`、`comparison.json`及子stage | 136.79秒、0更新、128条raw token/状态/输入/有效解码一致；仅覆盖这个控制，不证明所有训练行为正确 |
| Early100实际运行 | `early100/run/`、`early100/execution-summary.json` | 100更新/原892调度/45warmup、实际3200条ID、三个checkpoint完成标记、四份生成预测；campaign保持failed，other缺失 |
| Early100逐句与截断审计 | `early100/analysis/`、`early100/truncation-audit.json` | 三个fixed128有效比较；clean含截断的诊断比较明确comparison_valid=false；同ID排除仅作敏感性 |
| Early100结束与监控 | `early100/final-host-snapshot.json`、`early100/run/*/monitor/`、`early100/run/training/tensorboard/` | 实际GPU/磁盘、进程退出和Plex保留；100条训练与1条dev loss事件；原告警保留并解释 |
| Early100新增图片 | [figures/early100/README.md](figures/early100/README.md) | 两组PNG/SVG/PDF，显式标PARTIAL；输入/导出SHA、重复渲染一致、视觉检查；不绘制未通过的full-dev |
| 独立科研图片 | [figures/README.md](figures/README.md)、[figure_manifest.json](figures/figure_manifest.json) | 三组PNG+可编辑文本SVG+矢量PDF；逐图metadata保存所有输入SHA256、实际绘图数据摘要、输出SHA256和可复算命令 |
| 已提交阶段的跨机Git备份 | `early100-validation/prior-stage-backup.json` | 提交`7a085d3`的Git bundle已经verify且本地/远端SHA256一致；只覆盖已提交代码、文档、小报告和Git历史，不含音频/权重 |
| 交付完成性审计 | `DELIVERY_AUDIT.md` | 已完成项、证据路径、负向质量结果与尚未执行范围 |

## 大文件与远端恢复能力

音频、官方归档、HF模型缓存和完整恢复权重继续留在`/mnt/raid/asr-lab`；本地Git只保存代码、收据、预测、指标、事件与小型图表。没有把音频、HF缓存、大权重或凭证提交仓库。**RAID上的音频和权重尚无独立异地副本；代码/报告bundle不能替代大文件灾备。**

现有远端文件清单以[zero-update-control/final-host-snapshot.json](zero-update-control/final-host-snapshot.json)为证据，捕获时间为`2026-10-23T04:17:28.023693+00:00`。它记录两个恢复目录：

| 目录 | 快照中的文件合计 | 恢复相关文件 |
|---|---:|---|
| `/mnt/raid/asr-lab/runs/qwen-campaign-20261022-a/main-epoch1/checkpoint-800` | 24,473,489,956 bytes（约22.793GiB） | 两个model safetensors分片、索引、optimizer、scheduler、trainer_state、两rank RNG、processor/tokenizer、checkpoint metadata |
| `/mnt/raid/asr-lab/runs/qwen-campaign-20261022-a/main-epoch1/checkpoint-892` | 24,473,505,789 bytes（约22.793GiB） | 同上；这是完整恢复点，不只是模型export |

同一历史快照记录GPU0原有PersonaPlex进程仍在、GPU1没有ASR进程，以及磁盘余量。该快照仅作历史证据。新的[Early100结束快照](early100/final-host-snapshot.json)于`2026-10-23T04:58:22.612641+00:00`捕获：仅既有PersonaPlex占GPU、ASR全部退出，项目233.82GiB、空闲3078.27GiB。另保留25/50各约7.61GiB模型导出、100约22.79GiB完整恢复状态；100未另做恢复训练测试，详见Early100报告。

## 已有Git备份的范围

[early100-validation/prior-stage-backup.json](early100-validation/prior-stage-backup.json)记录提交`7a085d3dcfd146b3f2cde284bc2107d5a91edd7c`的bundle：本地`.git/asr-backups/asr-lab-before-early100-7a085d3.bundle`，远端`/mnt/raid/asr-lab/backups/asr-lab-before-early100-7a085d3.bundle`。收据显示`git bundle verify`通过，跨机SHA256均为`a6fb2a11dd230c862ee53d23f60f7d4c59678dd6c9ecb8d23865fbe629b862d6`。

它覆盖该提交及历史中的代码、文档和小报告；不覆盖后续未提交文件、raw audio、HF cache或远端恢复权重，因此不能称为整个实验已具备灾备。Early100结果提交对应的新跨机bundle及验签状态以[final-backup.json](early100/final-backup.json)为准；收据列出确切commit、大小、SHA256和备份范围，不把未执行other或未保存token加入已交付项。

## 图片与再现

从仓库根目录执行：

```bash
.venv/bin/python -m pip install -r requirements-plots.txt
.venv/bin/python scripts/plot_experiment_results.py
```

默认输入均为现有本地JSONL/JSON，不加载模型、不访问GPU、不生成新评测。每个图的源文件清单及完整SHA256位于同名`.metadata.json`；总命令、脚本SHA256、Python/matplotlib版本与输出SHA256位于`figures/figure_manifest.json`。

Early100已同步真实数据。复算新增两图：

```bash
.venv/bin/python scripts/audit_early100_partial.py
.venv/bin/python scripts/plot_early100_results.py --early100 reports/early100/run --partial-diagnostic
```

完整绘图模式仍拒绝本次failed状态；部分模式只画已通过固定128和已完成训练。`figures/early100/figure_manifest.json`保存全部输入与输出指纹，`render-verification.json`记录重复渲染六个导出SHA一致和审计收据。主训练原图只有8个实测dev-loss点，不虚构892步dev-loss；新训练图只有100步和1个实际dev-loss记录。

目前尚无独立test、人工听音、中文、960h或流式验证产物。对外描述应保持这些状态为未执行；也不能把更低teacher-forced loss或更高吞吐当成更好WER。
