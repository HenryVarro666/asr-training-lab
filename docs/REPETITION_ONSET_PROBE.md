# 重复输出出现阶段探针：原始模型与25/50/100步单句对照

状态：2026-10-23，执行前合同。用户已授权先规划预算与停止条件，再完成这一次小规模推理对照；本文件不代表已经执行。Astra负责合同、后续代码及结果只读审阅，Sol负责实现。完成本轮后停止，不自动接续实验或训练。

## 问题与已有证据

上一探针对Early100 checkpoint100运行两次原batch16和一次目标单句：三次目标输出均为512个token、无EOS，预解析文本为170次 `CULINARY` 加末尾 `CUL`，官方重复清理将其压成 `CULINARY CUL`。原始token已保存，批次与单句的目标token相同。这只证明该检查点在受测条件下的可重复现象，不能解释训练原因。

本轮问题是：**相同目标、相同单句输入与解码条件下，原始模型及现存25/50/100步检查点分别产生什么原始输出，最早在哪个已保存检查点观察到预算耗尽伴随重复？** 不将总体WER变化与单句循环混为一谈，不推断其他样本或开发集。

代码图查询已执行；本次query未覆盖目标，不能据此判断整个代码图为空。主代理对原探针模块的查询已取得13个节点。本审阅对未命中的新runner使用只读源码与上一轮singleton输入/配置记录作fallback核对。

## 四次固定调用

唯一目标为 `652-129742-0006`，沿用上一探针 `target-single` 的manifest row、音频、prompt及输入证据。

| 顺序/标识 | 权重来源 | batch | generate次数 |
|---|---|---:|---:|
| base | 本地缓存 `Qwen/Qwen3-ASR-1.7B`，revision `7278e1e70fe206f11671096ffdd38061171dd6e5` | 1 | 1 |
| step025 | Early100 `training/checkpoint-25` | 1 | 1 |
| step050 | Early100 `training/checkpoint-50` | 1 | 1 |
| step100 | Early100 `training/checkpoint-100` | 1 | 1 |

Early100根为 `/mnt/raid/asr-lab/runs/qwen-early100-20261022-a`。四个worker均为独立新进程，只使用物理GPU1、顺序运行，每个模型仅调用一次generate；合计 **4次调用、4条样本消费、1个不同样本、0次optimizer更新**。上一探针的step100结果不代替本轮fresh生成，也不构成追加第五次调用的理由。

不得新建optimizer、反向传播、训练、下载模型、复制大权重、加载第五个模型、执行其他样本或dev-other。两份model-only检查点足够用于推理，不需要其不存在的optimizer状态。

## 冻结执行条件与漂移门槛

保持旧singleton路径：官方wrapper/processor与parser版本、BF16、SDPA、eval/inference模式、16kHz音频、左padding、空system/user audio及English assistant前缀。不reset RNG、不切换TF32/确定性设置、不添加logits处理器、停止条件、scores输出或beam/sampling变体。调用必须保持：

```python
model.generate(**inputs, max_new_tokens=512,
               do_sample=False, num_beams=1, use_cache=True)
```

官方outer.generate自行传递EOS集合与return_dict参数，不能再重复传入。透明捕获最终thinker配置时原样调用/返回现有逻辑；不为获得trace改变generate kwargs或增加调用。

执行前保存并核验以下身份链：

1. 原始模型本地固定revision及已保存基础模型来源记录；三个checkpoint对应原Early100完成标记、权重分片/索引/config/tokenizer/processor/chat_template文件hash。每个模型有自己的预期权重身份，**不要求不同检查点权重相等**。25/50/100权重必须匹配历史完成标记SHA；base目前没有历史shard SHA，必须明确记为“固定revision路径与本轮首次捕获hash”，不能冒称完成了历史逐字节对照，也不能把首次捕获包装成已有来源证明。
2. 音频FLAC文件hash与decoded mono16k float32 PCM hash、目标manifest row；与上一probe singleton的ID/prompt/input IDs、有效长度、masks、音频特征及其dtype/shape/hash逐项对齐。上一记录的singleton prompt宽度及有效长度均为137；若不同先定位，不静默接受。
3. 软件版本、模型/processor/parser及复用helper源码hash、物理GPU映射、推理dtype/注意力和运行设置。模型特有的路径、权重hash、训练步数分别记录，不把预期权重差异误判为执行环境漂移。
4. 所有worker的实际生效generation规则必须与上一singleton及本轮base一致：明确kwargs、EOS/PAD/BOS实际tensor、logits processor类型及有效配置。raw配置完整保留；仅在已确认greedy模式后排除确实不生效的sampling字段和版本元数据。不能通过改写模型配置使对照“看起来一致”。

输入在generate前检查；有效配置如只能在首次调用内捕获，则该次产物保留，发现漂移后立即停止后续worker。不得等到四次全部跑完才将漂移标成普通观察。token、预解析文本或parser输出不同属于本轮要测量的结果，不能作为身份漂移停止条件。

### 有界权重校验I/O

只读预检已有4模型全量SHA读取超过240秒的证据；该操作没有运行模型。为避免重复扫描，本轮每stage仅对**当前模型**权重做一次完整SHA，纳入该stage的240秒，不预扫全部四模型，也不扩大预算。

权重身份由新supervisor负责：记录每个分片的逻辑路径、resolved path、device/inode、size、mtime_ns、ctime_ns与SHA；若为软链，同时记录指向。完整SHA前后核验stat相同，并在启动worker前和worker结束后再次匹配这些stat。25/50/100 SHA对历史anchor；base保存首次捕获的限制说明。任何缺失、替换或stat变化即停止。

旧worker的 `identity.files` 可只包含较小的源码/config/tokenizer/processor/receipt文件，权重SHA/stat放独立字段或文件，避免 `Campaign.execute` 和旧worker入口/末尾重复扫描大权重。报告不得声称worker另做了全量SHA，或结束时再次证明了所有字节相同。stat是本地非对抗性变更检测，不等同结束时重算密码学hash。

SHA必须分块检查deadline或放在受supervisor监管的owned子进程；不能用无中断的大文件读绕开240秒。SHA、加载、生成、写盘总耗时超出stage预算就fail-stop，不以“只是校验”延长时间。

## 原始证据与解析层分离

复用上一probe的完整取证格式：实际prompt、全部input IDs/masks/feature mask、所有输入tensor hash与dtype/shape、完整sequences、continuation IDs（含EOS后padding）、有效EOS/PAD位置、special token位置、两种skip-special解码、实际parser输入及输出、归一化文本、参考词和S/D/I、worker状态/时间及产物hash。

完整返回sequences必须先保存，再检查shape/prefix；异常不能毁掉取证对象。只在shape可解释时计算continuation。模型输出正常时继续使用原parser；可在其后对同一预解析文本调用原repeat-fix作CPU观察，但不得用它替换实际parser输出或影响模型生成。`skip_special_tokens=True/False` 均保持 `clean_up_tokenization_spaces=False`。

本轮不采集逐步logits，不声称测到了argmax margin或数值微扰。实际已观测的浮点输入通过现有fingerprint有限性检查；未观测scores则不声称验证其有限性。

## 预定义重复量化

量化由CPU读取保存的原始证据完成，不再调用模型。**不得只匹配 `CULINARY`，也不得把parser改动本身当作循环判据。** 同时报告以下三层：

1. **停止/长度**：返回宽度、首次EOS位置、旧规则截断与有效EOS规则截断；是否恰好512个新token且未见EOS。原始记录永不删改。
2. **原始token周期**：用于分析的token串取首次有效EOS之前的部分，排除EOS及其后padding；无EOS则用整个continuation。对所有起点扫描周期长度1至16，寻找同一token块连续精确重复至少2次的串联run。记录最长run的起点、周期块/长度、完整重复次数、覆盖token数及占分析串比例；另记录其中是否有至少20个完整周期且覆盖≥50%的run。末尾不完整周期另列，不能算成完整重复。并列按覆盖长度降序、周期长度升序、起点升序固定选择。
3. **预解析文本周期**：对实际 `parser_input` 以空白分词，保留原大小写/标点，不使用normalizer改变序列。按相同规则扫描词块周期1至8，记录最长run、重复次数、覆盖词数/比例及≥20周期且≥50%覆盖条件。直接保存匹配块，便于人工复核；不在parser压缩后的短文本上计算这项指标。

两个独立flag必须保留：`strong_periodic_repetition` 表示token层或词层存在至少20个完整周期且覆盖至少50%的run；`budget_exhausted` 表示512新token且无有效EOS。`budget_exhausted_with_periodic_repetition` 为二者的组合。即便模型出现强重复后提前EOS结束，仍必须保留strong=true，不能因没有耗尽预算漏掉重复观察。

在上述独立flag之外，本轮停止/组合分类预先固定为：

- `budget_exhausted_with_periodic_repetition`：512新token、无有效EOS，且token层或词层存在**至少20个完整周期、覆盖该层分析序列至少50%**的精确串联run。
- `budget_exhausted_without_detected_periodicity`：耗尽512且无EOS，但不满足上述重复阈值。
- `eos_terminated`：存在有效EOS；仍报告原始重复指标，不能因正常结束就隐藏长重复。
- `unexpected_stop_without_eos`：少于512且无EOS；保存事实并标为需要解释的停止异常，不宣称正常结束。若实际停止规则与冻结条件不符，按漂移门槛停止。

这是一套描述性、可复现的探针判据，不是语言错误或无限循环的普适定义。两个短词重复、合法修辞或普通转录中的少量重复不能触发预算循环分类。有限窗口扫描可能漏掉更长周期、近似重复、语义复述；“未检出”不等于不存在任何重复。有限512预算也不能证明增加预算后仍会永不结束。

## 预算与失败停止

| 项目 | 硬上限 |
|---|---:|
| 总墙钟 | 20分钟，含校验、4worker与CPU汇总 |
| 每worker | 240秒，含加载/一次生成/保存 |
| 生成调用/消费 | 4次 / 4条 |
| 本轮磁盘增量 | 1GiB，含run及本轮cache增长 |
| 项目总量 | ≤300GiB |
| 挂载盘空闲 | ≥500GiB |

阶段deadline取240秒与总剩余时间的较小值，不重置总预算。使用 `/mnt/raid/asr-lab` 下独立run、项目全局锁、独立进程组及10秒资源监控；不向根盘写cache。每worker启动前及迁入GPU前检查物理GPU1 compute apps；其他任务占用则停止当前启动，不驱逐、不自动等待/重试。GPU0 PersonaPlex不动。

身份/输入/有效配置漂移、OOM、超时、异常退出、非有限已观测浮点值、I/O失败、缺少必需证据或资源越界，立即停止剩余worker并保留部分状态。只清理本supervisor拥有的进程组。

**重复、截断、与上轮token不同本身是观察，不是失败重试理由。** 在身份一致的前提下继续下一预设模型；不修改512预算、阈值或prompt，不增加第五次调用。四次全按合同完成时仅称“推理对照完成”，不改变旧Early100的failed状态。

## 解释边界与报告

逐点列出base/25/50/100的EOS、token数量、token/词周期run、操作性分类、预解析及解析后文本、S/D/I，保留所有四个点，不只报道支持某个假设的检查点。

- 分别报告“最早已观测strong重复”与“最早已观测预算耗尽伴重复”。若25有strong但EOS正常结束、50才出现预算loop，两种最早受测点分别为25和50，不能合并成一个起点。
- 若base未观察到某一flag、25观察到，只能说“最早已保存且受测的该类异常检查点为25；base与25之间出现了受测行为差异”。用区间时仅条件性写作 `(0,25]`，不能把25称为确切首次异常更新，也不能断言此前每一步都正常。
- 若25无而50有，或25有/50无/100再有，严格逐点描述，不假定单调退化或阈值跨过后不可逆。未保存的更新不可插值推断。
- 单模型一次生成没有估计重复发生概率、训练seed方差或总体泛化；greedy也不保证CUDA跨运行位级一致。step100与上轮不同，应报告而非追加重试。
- 所有worker为fresh进程，未重放旧评测此前137个batch；本轮不能重建旧历史raw tokens。token证据可以定位本轮生成/解码/parser哪层产生表象，不能单独证明学习率、数据风格、padding、训练实现或灾难性遗忘的因果。
- base与微调检查点的对照有助于把现象与训练后状态变化联系起来，但没有单因素训练消融，不能据此确定训练中的具体原因。不把这一个样本的诊断WER作为全量精度指标。

## 实现范围与验收

仅新增本轮独立supervisor/worker入口、重复量化CPU helper及必要测试，最终报告写入新的 `reports/repetition-onset/`。stage标识固定为 `base`、`step025`、`step050`、`step100`，步数分别为0/25/50/100。可复用上一probe已验证的 `generate_and_persist_raw`、`capture_generation`、`decode_records`、GPU guard与指纹函数，不复制成另一套生成逻辑。

上一supervisor固定checkpoint100及3个stage，**不能直接放宽旧实验入口并覆盖旧产物**。新supervisor可以分别以四份已核验模型身份调用冻结worker的 `target-single` 模式，各自独立进程/目录，并在新收据绑定base/25/50/100角色；也可用新worker入口复用上述纯函数。旧冻结训练/评测文件、旧probe代码与报告默认保持不变。确需改共享helper时须单列差异、重新测试并冻结两个来源版本，不能悄然改变已审路径。

批准的新runner薄wrapper方案：在每个独立worker进程中临时包裹旧模块的 `generate_and_persist_raw`，读取已保存的 `inputs.json` 与 `metadata.json`，对上一singleton及本轮base核验共同输入、runtime和kwargs；通过后原样调用旧函数恰好一次，失败时实际generate调用数为0。入口用 `finally` 恢复旧函数binding与临时argv，记录wrapper源码hash和实际调用计数。不复制生成逻辑、不增加kwargs/processor、不reset RNG。有效配置由原捕获逻辑在该次首次调用内取得，stage完成后检查；不符即禁止下一阶段。

CPU契约至少覆盖：固定4模型/4调用/单一ID顺序；模型特定身份与共同输入/有效配置对齐；重复阈值与边界、EOS后padding不参与run、最长run并列规则、短重复不误报及非CULINARY重复；身份漂移在下一worker之前停止、预期截断不重试；完整raw保存顺序及全部必需证据。

Astra代码只读审阅无阻断、契约测试通过、源码冻结与GPU空闲条件满足后，主代理执行已授权的一次对照。计划完成后Astra停下，等待主代理明确followup触发代码审阅；不靠消息自动启动新工作。
