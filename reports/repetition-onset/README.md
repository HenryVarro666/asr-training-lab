# 四模型单句重复出现阶段对照

2026-10-23。**受测保存点中，100步首次观察到强周期重复与512-token预算耗尽；原始模型、25步、50步均正常EOS结束。** 本轮定位的是这条样本在已保存检查点上的行为变化，尚未确定具体首次异常更新或训练根因。历史Early100的failed状态保持不变，没有新增训练。

## 预先规划与实际投入

先按[合同](../../docs/REPETITION_ONSET_PROBE.md)固定预算、四次调用顺序及停止条件，Astra审阅、Sol实现。执行源码固定为`a1d1c892906cd586f16d2403a5e12ad3fccef202`；本地100项、远端14项CPU测试通过后启动。总上限1200秒、单stage240秒（含当前模型SHA、加载、生成与保存）、增量1GiB；未追加第五次调用、其他样本或训练。

四个独立进程依次加载原始Qwen3-ASR-1.7B及Early100的25/50/100步权重，仅用物理GPU1。唯一目标`652-129742-0006`，相同音频、137-token singleton prompt及输入tensor指纹，相同BF16/SDPA、greedy、num_beams1、use_cacheTrue、max_new_tokens512。原始配置文件差异原样留档；实际输入、运行环境与生效generation规则均通过对旧singleton和本轮base的一致性门槛。

实耗**99.16秒，4次generate、4条样本消费、1个唯一ID、0次optimizer更新**。run大小1,549,627 bytes（约1.48MiB），同口径项目增量1,578,979 bytes，低于1GiB。结束项目约233.63GiB、RAID余量约2587.13GiB；GPU1无残留计算进程、global lock释放，GPU0 PersonaPlex PID2154793继续运行。详细数字以[执行摘要](execution-summary.json)及[最终主机快照](validation/final-host.json)为准。

## 同一句的四个结果

下表token数包含终止EOS，EOS位置从0计数。S/D/I使用原固定normalizer、同一16词参考；这是单句诊断，不是全开发集WER。

| 模型 | token数 | 首EOS位置 | 强周期重复 | 512且无EOS | S/D/I | 原parser输出 |
|---|---:|---:|---|---|---|---|
| 原始 | 27 | 26 | 否 | 否 | 0/0/0 | 正确16词，大小写与标点自然 |
| 25步 | 35 | 34 | 否 | 否 | 0/0/0 | 正确16词，开头混合大小写、后段大写 |
| 50步 | 37 | 36 | 否 | 否 | 0/0/0 | 正确16词，全大写 |
| 100步 | 512 | 无 | 是 | 是 | 2/14/0 | `CULINARY CUL` |

[原始模型](run/base/target-single/decoded.json)、[25步](run/step025/target-single/decoded.json)、[50步](run/step050/target-single/decoded.json)、[100步](run/step100/target-single/decoded.json)均保存未缩略的解析前后文本及完整token。前3个模型去除正常EOS后的文本没有被重复清理改变。大小写逐渐变化是本次可见现象，不能据此把转录风格认定为循环原因。

## 重复位于生成序列，parser把它压短

100步的原始continuation开头为`[34,1094,32593,356,1094,32593,356,...]`。从索引1起，三token周期`[1094,32593,356]`连续完整重复170次，覆盖510/512个token（99.61%）；末尾还有未完成周期。全512个token没有有效EOS或其他特殊token。

本轮100步的完整sequences和continuation与上一探针的singleton逐token一致。两种skip-special解码均得到1533字符：170次`CULINARY`，再接`CUL`。按空白分词，单词周期覆盖170/171词（99.42%），远高于预设的20个完整周期且覆盖≥50%阈值。原官方parser再把这个长串压成12字符、两词的`CULINARY CUL`。因此短hypothesis隐藏了长重复；本轮证据不支持“parser凭空制造了重复”。生成原因与后处理表象分开记录，未修改parser作为所谓模型修复。

“强周期重复”与“512无EOS预算耗尽”独立计量，本轮两者最早的受测保存点恰好均为100。25/50未检出强重复；最后一个受测阴性点是50。**不能把100称为精确首发更新，也不能断言所有未保存的1–99步都正常或退化必定单调。** 只能说在50与100两个采样点之间，观察到由正常输出变为重复耗尽的差异。

## 身份、监控与独立复核

四模型每stage只做一次当前模型完整SHA，均在预算内；25/50/100与历史prefix-complete中记录的SHA匹配。base只有历史固定revision路径，本轮首次捕获shard SHA，不能声称有历史逐字节权重对照。SHA前后及worker前后stat稳定，属于非对抗变更检测，不是结束时再次完整重算SHA。先前一次只读全模型哈希预检超240秒、没有有效逐文件收据且reader已退出；该失败与实际99.16秒GPU对照分开记录，未把它算作一次生成或隐藏掉。

- [Astra代码审阅](validation/astra-code-review.md)、[部署哈希](validation/deployment.json)、[启动收据](validation/launch.json)。远端Git HEAD是旧部署基底；实际执行以本轮逐文件SHA和上述本地source commit为依据。
- 每个`run/<stage>/target-single/`保存输入IDs/masks/特征tensor及指纹、完整sequences/continuation、EOS/PAD位置、两种decode、parser输入输出、归一化S/D/I、实际生效生成规则、runtime、stdout、退出状态及10秒资源CSV/JSON。
- [完整产物清单](artifact-manifest.json)核验85个文件远端与本地逐字节一致；`.pt`也已镜像本地，但Git忽略它们。模型权重与原始音频未复制进Git。
- 独立CPU审计与Astra结果审阅在归档前完成，正式结论分别保存在[独立审计](independent-audit.md)和[结果审阅](validation/astra-results-review.md)。审计只读取保存证据、CPU tokenizer/parser和周期统计，不新增模型推理。

通用monitor的末次采样可能仍为running，且latest_checkpoint字段可能匹配inputs.pt/raw-tokens.pt；这些是取证tensor而非新训练权重。结束以worker/stage退出0、campaign complete、监督器退出0及最终主机快照为准。本轮只有四条推理观察，没有新loss曲线，历史训练图保持不变。

## 解释边界与停止状态

本轮按合同完成后停止，没有自动后续队列。前100步fixed128总体WER的退化已在25步出现，而这条目标的强循环只在100步被观察到；两者不是同一个“起点”，不能用单句正常输出否定总体早期退化。

没有采集逐token logits、梯度或训练中的反事实，也没有更多中间checkpoint。因此尚不能区分学习率、训练样本/目标风格、更新实现、参数漂移等训练机制。四次fresh singleton生成没有重放旧full-dev此前137个batch，不能恢复当时未保存的token历史；不估计发生概率或推广到其他样本。本轮未改512预算、模型架构、训练数据或学习率，也没有补跑other或独立test。

代码、文档、小型结果与审计的跨机Git bundle覆盖以[备份收据](final-backup.json)为准；取证tensor保留在本地和服务器的run目录。
