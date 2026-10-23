# 固定512-token截断诊断结果

2026-10-23。**三次预定探针全部完成，异常可重复观察：生成输出重复直到512-token上限，官方parser再把重复长串压成两词。** 这定位了本探针中的输出与后处理行为，没有解释训练为何产生这种输出，也没有修复模型。Early100旧campaign仍为failed，other未执行。

实验先按[合同](../../docs/TRUNCATION_PROBE.md)规划，由Astra审阅、Sol实现，源码固定为`37eb2e8c3099a601768b4b76dafc52856ce340be`。本地86项与远端14项CPU测试通过后执行。未加载其他模型、改变生成预算、追加训练或第4次生成。

## 实际范围与预算

唯一checkpoint为`/mnt/raid/asr-lab/runs/qwen-early100-20261022-a/training/checkpoint-100`，BF16、SDPA、原English prompt、left padding、greedy、num_beams1、use_cacheTrue、max_new_tokens512均保持。只使用物理GPU1；GPU0 PersonaPlex PID2154793保持运行。

原dev-clean按ID排序的batch137（索引2192:2208）运行两次，目标`652-129742-0006`位于批内11，再单句运行一次。三次独立进程共3次generate、33条消费、16个唯一ID、0optimizer更新。

| 调用 | 条数 | worker实耗 | 目标continuation／EOS | 解析前内容 | 原parser输出 |
|---|---:|---:|---|---|---|
| batch16-first | 16 | 28.81秒 | 512／无 | 170 × `CULINARY` + `CUL` | `CULINARY CUL` |
| batch16-repeat | 16 | 25.28秒 | 512／无 | 170 × `CULINARY` + `CUL` | `CULINARY CUL` |
| target-single | 1 | 23.13秒 | 512／无 | 170 × `CULINARY` + `CUL` | `CULINARY CUL` |

总runner墙钟**130.69秒**，低于1200秒上限；每worker低于240秒。run占15,823,430 bytes（15.09MiB），同方法du观察的项目增量也是15,823,430 bytes，低于1GiB上限。末态项目233.62GiB，RAID空闲3076.07GiB，满足300GiB项目上限和500GiB reserve。详见[执行摘要](execution-summary.json)。

## 生成、解码、解析的证据

1. **生成**：三次目标continuation逐token完全相同，均512个token，不含实际EOS集合151645/151643或任何特殊token。完整sequences与prompt前缀一致，continuation按实际padded prompt宽度切出；batch16的其他15条具有EOS，不能把共享512宽度误报为全部截断。
2. **解码**：同一目标token序列在`skip_special_tokens=False/True`下得到完全相同的1533字符文本：`CULINARY`重复170次，再接未完成的`CUL`（171个空白分词）。本次特殊token过滤没有改变目标文本。
3. **解析**：原`parse_asr_output(..., user_language='English')`先调用`detect_and_fix_repetitions`，将上述重复串压成12字符的`CULINARY CUL`。同版helper独立观察与实际parser返回一致。先前保存的两词hyp隐藏了长重复，不能把它当作模型只生成两词的证据。

实际token起始为`[34,1094,32593,356,1094,32593,...]`，频数为34×1、1094×171、32593×170、356×170。完整数组保存在每个stage的`raw-tokens.json`及`.pt`，没有删去EOS后的batch padding。

目标参考含16词；三次原normalizer下均为S=2、D=14、I=0、H=0，与旧失败记录一致。两次batch16的完整输入、有效生成规则、所有16条原始tokens和解析结果全部一致。单句也有完全相同目标tokens，因此**原16条batch组成不是此现象的必要条件**；这不能排除其他实现因素，也不能直接推断学习率、遗忘或训练数据是根因。

## 可信边界

旧事件没有保存raw tokens，本探针也未重放之前137个batch。这里复现的是相同样本/权重/预算下的截断与解析后异常，不能声称恢复了旧事件的逐token历史。没有采集逐步logits，不声称已核对argmax margin或证明数值原因。无人工听音、独立test、other评测、训练扩展或模型修复。

探针的`complete`表示三次取证执行完毕；旧Early100的零截断验收仍失败。其历史campaign文件SHA已与旧本地档案逐字节核对，保持不变。

## 保存、复核与监控

- [run汇总](run/summary.json)、[完整执行状态](run/campaign.json)、[重复身份检查](run/repeat-identity-check.json)。
- 每个stage均保存输入IDs/masks/prompts及tensor指纹、完整sequences/continuation、EOS/PAD位置、两种decode、原parser输入/输出、S/D/I、有效generation配置、runtime、stdout、stage退出码和10秒资源采样；[产物SHA256清单](artifact-manifest.json)。
- [Astra代码审阅](validation/astra-code-review.json)、[结果复核](validation/astra-results-review.json)、[独立CPU审计](independent-audit.md)、[真实官方parser的synthetic契约](validation/parser-contract.json)。Synthetic验证与本次真实生成证据分开保存。
- [启动前实际空闲检查](validation/launch.json)、[结束主机快照](validation/final-host.json)：GPU1无本实验残留进程、global lock释放、PersonaPlex仍在。实验启动前曾等待其他视频任务自然结束，没有停止它们。

监控的最后10秒采样仍可能写`running`；完成以worker/stage退出0、campaign完成和最终主机快照为准。通用monitor的`latest_checkpoint`字段会匹配本次`inputs.pt/raw-tokens.pt`，它们是取证tensor，**不是新训练checkpoint**；本轮没有创建模型权重或训练曲线。

全部取证tensor已保存在服务器并镜像到本地`reports/truncation-probe/run/`；`.pt`受Git忽略，Git与bundle只收代码、文档和JSON/日志等小型证据。不会公开推送权重、音频或凭据。跨机代码/报告备份的确切覆盖以`final-backup.json`收据为准。

## 本轮结束

到此停止，不重试、不改token预算、不补跑完整开发集或扩大训练。若要继续定位重复输出的训练机制，应另立一个单变量实验与预算；现有证据本身还不足以选择修复方案。
