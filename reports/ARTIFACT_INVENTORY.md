# ASR实验产物归档索引

本地工作目录：`/Users/caochao/Documents/ChatGPT/train`；远端资源根：`/mnt/raid/asr-lab`。代码、文档、实际收据、原始逐句预测、监控记录与图片分别归档，结构合理。此前缺少独立可导出的图片，本轮已补齐三组PNG/SVG/PDF及逐图来源指纹。没有下载新的模型/音频、重跑训练或公开推送。

本清单记录已有证据的保存位置；它不是新的远端实时审计。下一阶段early100已获授权，启动、进度与完成须看它自己的独立收据；当前默认图只覆盖原892步实验、800步诊断与已完成的E1–E4测量。

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
| 独立科研图片 | [figures/README.md](figures/README.md)、[figure_manifest.json](figures/figure_manifest.json) | 三组PNG+可编辑文本SVG+矢量PDF；逐图metadata保存所有输入SHA256、实际绘图数据摘要、输出SHA256和可复算命令 |
| 已提交阶段的跨机Git备份 | `early100-validation/prior-stage-backup.json` | 提交`7a085d3`的Git bundle已经verify且本地/远端SHA256一致；只覆盖已提交代码、文档、小报告和Git历史，不含音频/权重 |
| 交付完成性审计 | `DELIVERY_AUDIT.md` | 已完成项、证据路径、负向质量结果与尚未执行范围 |

## 大文件与远端恢复能力

音频、官方归档、HF模型缓存和完整恢复权重继续留在`/mnt/raid/asr-lab`；本地Git只保存代码、收据、预测、指标、事件与小型图表。没有把音频、HF缓存、大权重或凭证提交仓库。

现有远端文件清单以[zero-update-control/final-host-snapshot.json](zero-update-control/final-host-snapshot.json)为证据，捕获时间为`2026-10-23T04:17:28.023693+00:00`。它记录两个恢复目录：

| 目录 | 快照中的文件合计 | 恢复相关文件 |
|---|---:|---|
| `/mnt/raid/asr-lab/runs/qwen-campaign-20261022-a/main-epoch1/checkpoint-800` | 24,473,489,956 bytes（约22.793GiB） | 两个model safetensors分片、索引、optimizer、scheduler、trainer_state、两rank RNG、processor/tokenizer、checkpoint metadata |
| `/mnt/raid/asr-lab/runs/qwen-campaign-20261022-a/main-epoch1/checkpoint-892` | 24,473,505,789 bytes（约22.793GiB） | 同上；这是完整恢复点，不只是模型export |

同一历史快照记录GPU0原有PersonaPlex进程仍在、GPU1没有ASR进程，以及磁盘余量。**不能把该历史快照当成early100授权后的当前实时GPU状态。** 主代理另作live核对；此归档任务没有访问远端。

## 已有Git备份的范围

[early100-validation/prior-stage-backup.json](early100-validation/prior-stage-backup.json)记录提交`7a085d3dcfd146b3f2cde284bc2107d5a91edd7c`的bundle：本地`.git/asr-backups/asr-lab-before-early100-7a085d3.bundle`，远端`/mnt/raid/asr-lab/backups/asr-lab-before-early100-7a085d3.bundle`。收据显示`git bundle verify`通过，跨机SHA256均为`a6fb2a11dd230c862ee53d23f60f7d4c59678dd6c9ecb8d23865fbe629b862d6`。

它覆盖该提交及历史中的代码、文档和小报告；不覆盖后续未提交文件、raw audio、HF cache或远端恢复权重，因此不能称为整个实验已具备灾备。early100最终报告提交后的新bundle尚待主代理另建，不能预先计为完成。

## 图片与再现

从仓库根目录执行：

```bash
.venv/bin/python -m pip install -r requirements-plots.txt
.venv/bin/python scripts/plot_experiment_results.py
```

默认输入均为现有本地JSONL/JSON，不加载模型、不访问GPU、不生成新评测。每个图的源文件清单及完整SHA256位于同名`.metadata.json`；总命令、脚本SHA256、Python/matplotlib版本与输出SHA256位于`figures/figure_manifest.json`。

后续early100产生并同步真实`metrics.jsonl`后，可用`--additional-run 'early100=实际run目录' --output reports/figures/with-early100`添加loss/LR/dev-loss曲线。未同步的阶段不补造曲线或WER点。主训练图只有8个实测dev-loss点，不虚构892步的dev-loss终点；WER图的连线只是三个测得checkpoint之间的视觉引导。

目前尚无独立test、人工听音、中文、960h或流式验证产物。对外描述应保持这些状态为未执行；也不能把更低teacher-forced loss或更高吞吐当成更好WER。
