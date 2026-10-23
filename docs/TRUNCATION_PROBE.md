# 截断样本探针：固定checkpoint100、固定512-token预算

状态：2026-10-23，执行前实验合同。用户已授权按此小规模预算规划后执行；本文本身不代表探针已经运行。Astra负责合同与代码审阅，Sol实现独立worker/supervisor及必要契约测试。禁止扩展训练或自动追加实验。

## 问题与已有证据

Early100在完整合格dev-clean生成中，对 `652-129742-0006` 保存了截断标记，解析后文本为 `CULINARY CUL`。旧evaluator没有保存原始token或解析前文本，所以不能独立重建其停止过程。旧campaign按零截断门槛失败，dev-other未执行。

本探针只回答三件事：在同一批组成下能否再次观察到截断；两次独立进程是否一致；改为目标单句后现象是否改变。保存原始token、EOS和实际parser输入输出，区分模型生成、特殊token解码与解析裁剪。它不修复模型、不证明退化原因，不改变旧campaign的 `failed` 状态。

已优先查询代码图，但图为空、未索引 `qwen_evaluate.py`，因此使用冻结源码与 `reports/early100/truncation-audit.json` 只读核对生成路径。

## 固定模型、数据与三次调用

唯一模型为远端 `/mnt/raid/asr-lab/runs/qwen-early100-20261022-a/training/checkpoint-100`。使用现有权重，不复制或下载新模型、不初始化optimizer、不执行反向传播，不加载原始模型或其他检查点。

先核验旧部署/源文件指纹、checkpoint100完成标记及权重/配置/processor文件hash、原dev-clean manifest与音频身份。记录本探针代码hash、运行版本及实际GPU映射。任何身份不符即停止，不能更新期望hash以求通过。

原评测在读取manifest后按utterance ID排序。使用**相同有序样本序列**，取索引 `[2192:2208]`，即0-based batch137。目标全局索引2203、批内索引11。必须与旧截断审计中的16个ID逐项一致，不得随机采样、重排或用manifest原始行号代替排序后的索引。

| 预定阶段 | 独立进程 | 样本与顺序 | generate调用数 |
|---|---|---|---:|
| batch16-first | 是 | 原batch137的16条，保持原顺序 | 1 |
| batch16-repeat | 是 | 完全相同16条，重复属于预定设计，不是失败重试 | 1 |
| target-single | 是 | 仅 `652-129742-0006` | 1 |

总共仅3次generate、33条样本消费，实际不同样本总数16。本轮所有worker只使用**物理GPU1**，GPU0现有PersonaPlex保持原状；不运行dev-other或完整开发集。

每次从同一checkpoint新进程加载，BF16、SDPA、eval/inference模式、相同软件版本、相同左padding、16kHz音频、相同空system/user audio prompt与 `language English<asr_text>` 前缀。生成调用保持旧evaluator语义：

```python
model.generate(**inputs, max_new_tokens=512,
               do_sample=False, num_beams=1, use_cache=True)
```

官方outer.generate仍自行设置 `return_dict_in_generate=True` 并向thinker传递EOS停止集合。不能重复传这个参数，不能把不同默认值当作未使用的配置。记录thinker最终生效配置及明确kwargs，保留原始配置与对当前greedy有效的配置；忽略字段必须有明确的“不参与当前模式”依据。

**不得为了观察而加入 `output_scores`、`output_logits`、新的logits processor、stopping criteria、beam/sampling设置、额外generate或前缀续跑。** 如用透明wrapper捕获最终配置，必须原样调用/返回，不能改变processor顺序、scores、RNG或停止规则。默认本轮不采集逐步logits；未采集就不能声称已核验其有限性或argmax margin。

## 必须保存的证据

每个worker保存独立目录与原始完成/异常状态，不覆盖旧输出。最少包括：

- 样本ID顺序、manifest/audio hash、duration、batch位置；模型与processor身份、软件版本、GPU物理映射、TF32/确定性设置及实际调用参数。
- 处理后的input tensor名称、dtype、shape、完整input IDs与attention mask；feature attention mask和浮点音频特征hash。区分**批补齐后prompt宽度**与每句有效prompt token数，记录切除continuation的准确边界。
- 完整返回sequences与逐句continuation的token IDs，**保留EOS之后的batch padding**。不得只保存去掉特殊token后的文本或截至首次EOS后的canonical tokens；可另外保存canonical视图。
- 逐句continuation宽度、EOS151645/151643的全部位置及首次位置、PAD ID与其位置、前后特殊token分布、实际旧规则计算出的截断标记。EOS/PAD如有重合，须按配置和序列位置说明，不据token ID武断判断哪些是补齐。
- 同一continuation的 `skip_special_tokens=False` 与 `True` 两种解码，均保持 `clean_up_tokenization_spaces=False`；旧路径实际使用的parser输入，即skip-special=True结果；调用同版 `parse_asr_output(..., user_language='English')` 的完整输出。记录归一化文本时继续使用原normalizer，不能把后处理变化混进模型变化。
- 保存所有16条的结果，不只目标。两次batch16比较完整输入hash、有效生成配置、原始token IDs以及parser输入/输出；target-single单独记录输入shape变化，不能要求它与batch16的补齐输入hash相同。
- worker耗时、阶段退出码、10秒资源监控、异常日志；最终小型CPU汇总附所有输入/产物hash。若worker在generate途中被终止，标记原始输出缺失，不能填造token记录。

“宽度512且无识别EOS”是旧规则下的截断观察；每句返回张量宽度包含batch补齐。必须同时呈现EOS位置，不能把共享512宽度当成16句都生成512个有效token。重复token/特殊token现象可做描述性计数，但不能仅凭两词解析文本命名循环类型。

## 资源预算与停止条件

| 项目 | 上限 |
|---|---:|
| 整个探针墙钟 | 20分钟，包括身份校验、三个worker、写盘和CPU汇总 |
| 每个worker | 4分钟，包括模型加载、一次generate与证据保存 |
| 总生成消费 | 3次调用、33条样本；不能追加第四次 |
| 磁盘增量 | 1GiB，计入run目录与本轮造成的项目cache增长 |
| 项目总量 | ≤300GiB |
| 挂载盘空闲 | ≥500GiB |

每阶段有效deadline取阶段上限与总剩余预算的较小值；不用剩余预算追加任务。所有路径使用 `/mnt/raid/asr-lab`，不向系统根盘写缓存。模型已在本地，禁用网络下载；不创建权重副本。复用项目全局锁和独立进程组，执行前检查GPU1没有本项目遗留任务及足够显存。

**预期截断本身是本探针要记录的观察，不作为异常重试理由，也不阻止下一个预定探针。** 三个预定worker全部按合同结束时，可以称“探针执行完成”；不能称“旧评测通过”或“截断已修复”。

OOM、模型/输入身份漂移、非有限观测值（若实际观测到scores，允许合法mask产生的负无穷，不接受NaN/正无穷或无有限候选）、超时、I/O失败、预算越界、异常退出或缺少必需证据，立即停止剩余worker并保留部分状态。不自动重试、不改batch、不放宽token预算、不修改异常文本。只清理本supervisor拥有的进程组，不操作PersonaPlex或其他会话。

## 解释与结束规则

- 两次batch16均重现相同token与截断：支持此输入组成和实现下可重复观察。还需据原始token/解析证据判断现象发生在哪一层；不自动推断训练机制。
- 两次batch16不同：记录运行间差异；greedy不保证CUDA每次位级相同，未采集logits时不能声称已经证明数值微差是原因。本轮不临时追加trace或重试。
- batch16截断而单句不截断：支持对batch条件敏感；单句改变了padding、shape和计算路径，不能直接命名padding bug。单句结果不替代旧全体评测。
- 本轮不复现：只说明这三次条件下未观察到。新进程直接处理原batch137，**没有重放此前137个batch**；旧进程上下文和数值执行历史不同，不能因此宣布旧标记错误或原事件不存在。
- raw token可见大量特殊token但parser只留下短文本：同时展示两种decode与parser输出，描述各处理步骤丢弃了什么；不能仅凭长度变化宣布模型已正确识别音频。

结束后给出三次调用的事实、可确定与不可确定的部分及预算实耗。旧Early100报告、failed状态、clean诊断值和other未执行记录保持不变。任何后续修改解码、加载其他权重、跑other或重训，均不属于本轮合同。

## 允许的实现范围

建议新增 `src/asr_lab/qwen_truncation_probe.py`（单worker取证）、`scripts/run_truncation_probe.py`（有界顺序supervisor）和对应CPU契约测试；最终证据写入独立run与新的 `reports/truncation-probe/`。实现文件命名可按仓库惯例调整，但职责和界限不变。

复用冻结评测模块中的prompt/audio/模型加载与解析语义，固定源码hash；不修改 `qwen_train.py`、`qwen_data.py`、`qwen_evaluate.py`、`benchmark.py`，不修改旧实验报告、checkpoint或其他计划。契约测试至少覆盖：原batch索引/顺序、33条/3调用上限、EOS后padding保留、两种decode与实际parser输入、捕获逻辑不改变generate kwargs、超时/异常停止后续worker、完整/部分状态明确区分。Astra只读review通过、相应CPU测试通过且代码冻结后，再由主代理执行已授权的一次探针。
