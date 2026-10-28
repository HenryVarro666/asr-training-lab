# 双 H100 ASR 首轮实验结果

> 2026-10-27 仓库维护：项目名称更新为 **ASR Training Lab**，GitHub 地址为 [HenryVarro666/asr-training-lab](https://github.com/HenryVarro666/asr-training-lab)。本次核对了相关会话与保存结果，更新项目说明和后续计划；没有新增训练、模型调用或远程操作。以下实验数字与失败状态保持原样。新的[双 H100 计划书](docs/plans/README.md)均为待执行方案。

> 执行方式更新（2026-10-22）：用户已取消耗尽token/usage的目标，并明确继续受控实验。后续采用[分阶段诊断与实验](docs/NEXT_STEPS.md)：每阶段限定问题、预算和验收条件，依据结果安排下一步。

**首轮训练和评测已完成，但识别质量退化，当前微调模型不应替换预训练基线。** 首轮报告时间为2026-10-22T17:45:23.520665+00:00；当时遇到的用量限制属于历史状态。用户授权的Early100已完成100次更新，随后按评测质量门槛停止；最新阶段见下方。

## 后续诊断进展

**当前：[CPU数值精度修复与全2304行验收通过](reports/token-probability/cpu-repair-20261026-r1/REPORT.md)**（2026-10-26）：先规划后执行，仅复用12份原矩阵，确认FP32绝对logsumexp相减丢失精度。改为FP64派生统计后，原容差独立审计全部通过；目标概率最大误差1.52434e-13，原门槛1e-6。三阶段CPU累计129.004秒，50专项测试通过、130原文件SHA保持，三图与完整新旧证据已保存；新增模型／GPU／训练调用0，没有进入B。原GPU0 A执行完成和原数值审计失败记录均保持历史原貌，未自动恢复Demo。此修复没有解决或定位训练导致重复的原因。

**此前：[四检查点重复对照已完成](reports/repetition-onset/README.md)**（2026-10-23）：同一句音频、同137-token输入、同512-token生成预算，base/25/50/100各一次，共4次调用／0更新／99.16秒。base/25/50正常EOS结束且原normalizer下16词全对；100生成170次`CULINARY`加`CUL`，512无EOS，parser压成两词。最早受测阳性为100，上个受测阴性为50；不能定位精确首发更新或训练原因。独立CPU审计通过，85份run产物跨机一致，GPU1退出、Plex保留，旧Early100 failed未改，无追加训练。

**此前：[512-token截断探针已完成](reports/truncation-probe/README.md)**（2026-10-23）：固定checkpoint100与原生成规则，GPU1上原batch16两次、单句一次，共3次调用／33条消费／0次更新，用时130.69秒。三次目标raw token完全一致，均512个且无EOS；解码为170次`CULINARY`加`CUL`，随后官方parser压成`CULINARY CUL`。单句同样复现，原batch组成不是必要条件；训练根因尚未确定。raw tokens、EOS、解析前后文本、指纹、监控及独立审计已保存，旧Early100 failed状态不变，没有扩大训练。

已完成第一阶段CPU诊断：5551条预测重新标准化和S/D/I复算全部一致，配对统计确认开发集上退化；独立模型链条审阅和两条不等长音频的CPU processor对照未发现已证实的标签/解码错误。详见[诊断报告](reports/diagnosis/README.md)。

已完成一个零更新实验：现存checkpoint800的完整合格dev-clean/dev-other生成评测，仅GPU1，实际7.99分钟（上限20分钟）。WER分别为2.4192%和5.0418%，均差于原始模型，也略差于第892步。两集800→892的配对差值区间覆盖零；退化在第800步时已存在，未支持最后92步造成主要退化的假设。

运行目录：`/mnt/raid/asr-lab/runs/qwen-checkpoint800-diagnostic-20261022`。逐句预测、配对统计和资源记录已同步到本地。

随后[零更新保存／重载控制](reports/zero-update-control/README.md)已通过：用时136.79秒，仅GPU1、0次更新；两侧实际参数／输入／有效生成规则与128条raw token完全一致，WER同为1.5835%。Astra独立复核无阻断项。它没有解释训练退化，也不证明优化器更新逻辑全部正确。

**[Early100训练完成、评测部分完成后停止](reports/early100/README.md)**：保持892步调度和45步warmup，在两张H100上完成100次更新；LR、样本数及音频时长逐步核验通过，25/50模型导出及100完整恢复状态已验。固定128的25/50/100步WER为4.1997%/3.7177%/3.2014%，均差于原始1.5835%，说明最早观测25步时已出现退化。

完整clean生成2694条，但1条达到512-token预算；WER3.3317%仅作包含截断的失败诊断。runner保持failed并停止，other未执行。实际641.52秒，训练阶段279.29秒；run38.02GiB、项目233.82GiB。主机终态已确认无ASR进程、原PersonaPlex仍在，RAID空闲约3.01TiB。失败证据、逐句预测、配对统计、TensorBoard和PNG/SVG/PDF已归档，没有继续训练队列。

## 完成的工作

- Astra 6 完成方案与训练流程审阅；GPT-6.1 Sol 完成训练、数据处理和实验编排代码。
- 首轮训练代码固定为 `cf30f45e0beb005f43520c33e2ddad73ad67a01e`；该次远端运行前34项测试及5个subtests通过。新增控制实现为 `c1d8ac3`，原训练／数据／评测／benchmark四文件指纹不变；控制前本地完整60项测试、远端11项专属测试通过。
- 数据/模型/环境/日志/checkpoints全部位于 `/mnt/raid/asr-lab`。准备LibriSpeech train-clean-100共28,539条、100.591小时，及独立dev-clean/dev-other。
- 主模型 `Qwen/Qwen3-ASR-1.7B`，revision `7278e1e70fe206f11671096ffdd38061171dd6e5`；加载后唯一参数实测2,038,052,480。此前HF元数据估算2.349B不能作为实际训练参数数目。
- 主训练双卡DDP、BF16 autocast、FP32权重/梯度/Adam状态、SDPA、activation checkpointing；microbatch8 × 2卡 × accumulation2，总batch32；lr1e-5，一轮。
- 主训练完成892次optimizer更新、epoch=1.0，用时25.65分钟。含4个效率实验和完整评测的campaign总时长53.87分钟。
- 覆盖口径：训练集有28,539条唯一音频/100.59088小时，日志累计28,544次样本消费/100.60493小时，多出的5次与双卡批次补齐一致；未记录逐条曝光ID，不声称每条恰好消费一次。见[覆盖记录](reports/data/training-coverage.json)。
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

微调后dev-clean从871个词错误增加至1282个，dev-other从1575增加至2545个。主要增加的是替换和插入错误。训练平均loss约0.0658不能代替识别质量结论。首轮此处仅记录描述性差异；后续paired bootstrap与固定样本审阅另见诊断报告，不据此确定退化原因。

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

数据归档／manifest校验收据已补齐到[本地数据记录](reports/data/README.md)，Whisper完整合格clean预测已同步并逐句复算至[参考验证](reports/baselines/whisper-full-clean/verification.json)。TensorBoard事件文件也已同步，解析确认892条训练loss与8条dev loss，见[事件验证](reports/monitoring/tensorboard-verification.json)。

首轮campaign结束后，15分钟自动化 `h100-asr` 已删除。两项后续零更新诊断均由独立runner监控并设置20分钟期限，已完成退出。新增Early100以90分钟总预算、20分钟训练及每评测10分钟执行，在641.52秒时因截断质量门槛停止，未恢复无限自动实验队列。监控不重启训练、扩大预算或停止其他服务。

## 未完成的后续工作

生成错误、现存checkpoint诊断和零更新保存／重载控制均已完成，完整证据见[交付核对表](reports/DELIVERY_AUDIT.md)。Early100及新增截断探针已保存早期退化、生成重复至预算和parser压缩的证据；下一阶段仍需单独规划，尚未补做other、改变lr/更新范围或重训。`docs/ADAPTIVE_EXPERIMENT_POLICY.md`仍为历史策略草案，不恢复无限队列。新的完整训练轮次、中文/混合语料、流式评测、人工听音和独立test评测均未执行。
