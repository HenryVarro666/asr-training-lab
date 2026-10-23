# Early100：保持原892步调度的早期轨迹诊断

实验日期2026-10-22（Chicago），归档2026-10-23。**100步双卡训练完成；评测部分完成后，因clean出现1条生成截断而停止。** 三个固定128评测通过，完整clean留下诊断结果，other未执行；整个campaign状态保留为`failed`，不能称为五项评测全部完成。

已通过验收的固定128支持：**在最早观测的25步时已有退化，100步仍差于原始模型。** 不能据此确定首次退化发生在哪一步、解释因果，或推断other。默认保留原始模型，不部署本次微调权重。

## 预先固定的预算与实际执行

[原合同](../../docs/EARLY100_EXPERIMENT.md)在启动前固定了原始权重、完整训练manifest、seed42、两卡DDP、microbatch8×2卡×accumulation2=32、lr1e-5、892步linear调度及45步warmup。仅在100次更新后计划停止，没有用`max_steps=100`重建调度，也没有截短训练数据。

| 项目 | 预算 | 实际结果 |
|---|---:|---|
| 总实验墙钟 | 90分钟 | 641.52秒（10.69分钟）后触发评测质量门槛；未耗尽预算 |
| 双卡训练阶段，含加载/保存/退出 | 20分钟 | 279.29秒（4.65分钟），100次更新，exit0 |
| 单项生成评测 | 每项10分钟，原计划5项 | fixed25/50/100分别26.23/22.71/22.03秒；clean250.57秒；other未启动 |
| 本轮增量 | 80GiB | run实际占38.02GiB，未触发增量限制 |
| 项目/磁盘 | 项目≤300GiB，空闲≥500GiB | 结束项目233.82GiB，挂载盘空闲3078.27GiB |

实际run：`/mnt/raid/asr-lab/runs/qwen-early100-20261022-a`。两卡用于训练，生成评测按原计划顺序使用GPU1。未自动重试、放宽512-token预算、继续到892、启动学习率扫描或追加架构。

[执行汇总](execution-summary.json)、[campaign收据](run/campaign.json)及各stage原始命令、时间和退出码均保留。HF在提前停止时输出的整体train_samples/steps_per_second使用完整epoch分母，**不用于报告本轮真实吞吐**。

## 固定128：三个检查点均通过对齐与零截断条件

同128个样本ID、2905参考词、固定Whisper EnglishTextNormalizer、greedy、batch16、512-token预算。所有raw reference/解析后hypothesis重新标准化后逐条一致；S/D/I与原保存指标一致。

| 模型/更新步 | WER | S/D/I（总错） | 比原始增加pp | 差值95%区间pp | 全大写输出 |
|---|---:|---|---:|---|---:|
| 原始模型 | 1.5835% | 32/9/5（46） | — | — | 0/128 |
| 25 | 4.1997% | 96/13/13（122） | +2.6162 | [1.9713, 3.2963] | 125/128 |
| 50 | 3.7177% | 84/8/16（108） | +2.1343 | [1.4295, 2.9123] | 128/128 |
| 100 | 3.2014% | 73/9/11（93） | +1.6179 | [1.0396, 2.1997] | 128/128 |

按40名说话人配对bootstrap、seed20261022、10000次，重采样后重新计算总错误/总参考词。属于同一开发集上反复查看的探索性诊断，未作多重比较校正，不覆盖训练seed差异，也不是独立test确认。每份固定128比较的`comparison_valid=true`；完整campaign不因此变成通过。

25→50→100的WER点估计逐步回落；不能写成误差随训练步数持续增长。25步已有明显大写风格变化，但normalizer消除大小写，风格变化与WER变化不能直接建立因果。

## 完整clean：生成结束，但零截断门槛失败

2694条合格短音频、54027参考词全部产生输出；生成进程exit0。预先固定的零截断验收发现1条达到512-token预算，因此runner停止，**dev-other没有执行**。

| 已保存的clean诊断值 | 原始模型 | 第100步 |
|---|---:|---:|
| WER（含截断输出） | 1.6122% | 3.3317% |
| S/D/I | 643/157/71 | 1463/149/188 |
| 总词错误 | 871 | 1800 |
| 截断条数 | 0 | 1 |

这组数值仅作失败诊断。对应[配对文件](analysis/step100-dev_clean.diagnostic-paired.json)保留`comparison_valid=false`及三条截断相关原因；不把其中计算出的区间当成通过质量门槛的完整开发集结论。

异常ID为`652-129742-0006`，参考内容为花椰菜食谱。原始模型16个参考词全部正确；100步模型保存的解析后文本为`CULINARY CUL`，计S2/D14/I0。同batch其他15条未标截断。冻结evaluator逐句检查continuation长度与EOS；仅有batch补齐长度不足以触发标记。但本次没有保存解析前原始生成文本与token IDs，不能独立重建EOS/实际token长度，也不能断言是哪一种生成循环。

该句贡献16词错误，只占新增929错的1.72%。同时从两侧剔除该ID后，2693条/54011参考词的诊断WER仍为3.3030%对1.6126%。这是事后配对排除敏感性，**不替换原值、不恢复验收通过状态**；它说明该句不能解释其余词错误差异。详见[截断审计](truncation-audit.md)与[机器记录](truncation-audit.json)。

## 训练轨迹与保存检查点

[轨迹验证](run/training/trajectory-validation.json)通过：100步LR最大差0，每步样本数及音频时长完全匹配原记录；消费3200条唯一ID、11.31386小时。新run实际ID按rank/microbatch逐步保存。首轮没有逐条ID日志，因此同配置、seed和聚合量不能证明旧样本顺序完全一致。

两次loss/梯度并非位级一致：最早loss差异在step3，最大绝对差0.0081、平均绝对差0.002192；梯度最早差异在step2。新100步loss0.0407、原100步0.0474。差异在首次25步导出之前已出现；本实验不能据此归因为导出，也不声称严格复原了旧权重。

第100步更新使用LR为9.36245572609209e-6，scheduler推进后LR为9.350649350649352e-6，仍按原892步线性调度。`prefix-status.json`为`completed_bounded_prefix`且`completed_epoch=false`，与原训练器通用`interrupted`状态并列保存。

| 远端training下目录 | 已验证文件合计 | 恢复能力 |
|---|---:|---|
| checkpoint-25 | 8,168,279,273 bytes | FP32模型与processor/config；model-only，不可完整恢复优化器轨迹 |
| checkpoint-50 | 8,168,283,556 bytes | 同上 |
| checkpoint-100 | 24,473,368,342 bytes | 模型、Adam、scheduler、trainer_state、两rank RNG；完整恢复状态 |

合计以完成标记中所列文件为准，不包含标记文件自身。全部文件哈希、结构、步数、dtype和导出RNG控制已验；本次没有再启动恢复训练测试。检查点及原始音频仍仅在服务器RAID，不在本地Git或异地权重备份中。

## 代码、监控和归档

Astra审阅合同和实现，Sol完成实现；本地72项、远端12项测试通过。[部署凭证](../early100-validation/deployment.json)确认本地提交`3f4955b`与远端源文件SHA256一致。远端checkout仍记录旧HEAD且dirty，实际运行身份由部署清单和run源文件指纹界定；原训练/数据/评测/benchmark四文件未改。

TensorBoard已同步并解析100条train loss/LR/grad、1条step100 dev loss；无伪造892完成点。资源每10秒记录，训练期间两张H100都工作。四次通用`training_interrupted`告警对应计划100步停止；真正终态失败是clean截断质量门槛。没有OOM、非有限loss/梯度或磁盘阈值失败。完整评测期间的进度来自实际样本完成数。

[最终主机快照](final-host-snapshot.json)记录全部本项目ASR进程退出、tmux会话结束；GPU0原PersonaPlex PID2154793仍在，GPU1无本项目进程。TensorBoard本地转发检查HTTP200：`http://127.0.0.1:16006/`，连接可用性会随SSH会话变化。

新图片只展示已通过的固定128与实际100步loss/LR，不画被门槛拒绝的完整clean或未执行的other：见[图片索引](../figures/early100/README.md)。全部图片有PNG/SVG/PDF、输入hash与复算命令。代码、计划、预测、状态、事件、分析和图片的保存边界见[总归档索引](../ARTIFACT_INVENTORY.md)。

## CPU复算与下一步边界

```bash
.venv/bin/python scripts/audit_early100_partial.py
.venv/bin/python scripts/plot_early100_results.py --early100 reports/early100/run --partial-diagnostic
.venv/bin/python scripts/analyze_asr_predictions.py \
  --baseline reports/baselines/qwen-pinned-b16/dev-clean-fixed128.jsonl \
  --candidate reports/early100/run/evaluation/step100-fixed128/predictions.jsonl \
  --manifest reports/early100/inputs/dev-clean-fixed128.jsonl \
  --bootstrap-unit speaker --replicates 10000 --seed 20261022 --require-valid \
  --output reports/early100/analysis/step100-fixed128.paired.json \
  --details reports/early100/analysis/step100-fixed128.edits.jsonl
```

完整五项审计及完整绘图模式仍拒绝本次部分失败状态。没有为出图把campaign改为complete。

本阶段按门槛停止。下一步应单独规划这一异常样本的生成诊断，保存解析前token及停止条件，再决定是否补做other；这只是后续候选，不是已启动队列。当前没有足够证据归因为学习率过大、目标大小写、过拟合或灾难性遗忘；没有独立test、人工听音、中文或流式验证。
