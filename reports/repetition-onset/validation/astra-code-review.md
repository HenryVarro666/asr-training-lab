# Astra：四模型重复出现阶段探针代码审阅

日期：2026-10-23。状态：**最终代码审阅通过，无阻断项。** 已核对最终补充实现及本地100项测试通过日志；可在源码冻结、远端对应CPU检查与实际GPU空闲条件满足后，进入用户授权的一次有界对照。此文不是已运行GPU的结果报告。

## 范围与方法

- `scripts/run_repetition_onset.py`
- `src/asr_lab/repetition_metrics.py`
- `tests/test_repetition_onset.py`
- `reports/repetition-onset/inputs/model-reference.json`
- 本轮合同及已审冻结probe/helper的调用关系。

先调用图查询；本次query未命中新runner，因此fallback读取相关源码。主代理另有原probe模块13节点的图结果，不能将本query的未命中外推为全局没有索引。

没有修改实现代码、启动GPU、访问远端或提交Git；仅修改本人合同与此审阅文件。额外执行了纯CPU独立穷举对照：对1580组长度1–79的有限随机序列，将最长精确重复run实现与按全部起点/period枚举的参考逻辑比较，span/period/start/count及并列规则全部一致。该检查未加载torch模型或GPU，不替代完整契约测试。

## 已核对条件

1. 顺序为base/step025/step050/step100，每stage独立进程复用冻结worker的 `target-single`，没有复制generate实现或增加kwargs。薄wrapper读取保存的输入/runtime证据，漂移时旧generate函数尚未被调用；通过后只允许一次调用，finally恢复函数binding和argv。
2. 共同输入与运行设置对上一singleton及本轮base比较；实际effective config/special-token tensors/logits processor类型在stage验收时比较，异常会阻止下一模型。token/parser差异是观察，不被误作身份失败。
3. 当前模型的权重SHA只做一次，阶段240秒从SHA前开始，逐块检查剩余时间；大权重不进入旧worker和Campaign重复扫描的 `files`。25/50/100对应历史完成标记，base无历史shard SHA被明确标作本轮首次capture，不冒称历史字节比对。
4. SHA前后和worker前后核验resolved path/dev/inode/size/mtime_ns/ctime_ns；这属于本地非对抗变更检测，不等于结束时再次全量SHA。最终版已补充logical path、is_symlink、readlink指向及symlink lstat，并保存各文件SHA耗时。
5. 继承总1200秒、单stage240秒、1GiB增量、300GiB项目与500GiB空间保留门槛；使用全局锁、GPU1空闲检查、owned进程组退出清理。超时或失败不会自动重试/增加模型；GPU0 PersonaPlex未进入操作范围。
6. 原始sequences先保存后校验；保留EOS后padding与双decode/parser证据。量化时仅分析首次EOS前tokens，原证据不删改。token period1–16、词period1–8；最长run并列规则符合合同，末尾partial不计完整周期。
7. strong repetition（至少20周期且覆盖≥50%）与budget exhaustion分别保存，预算loop为二者组合；EOS提前结束的strong仍可被报告。总结分别给出最早已观测strong与loop，保留positive→negative反转，不给确切首次更新或训练因果结论。

## 最终测试及冻结指纹

已读取最终 `reports/repetition-onset/validation/local-unit-tests.log`：100 tests，3.832秒，OK。Sol报告本轮14项focused契约通过；审阅已核对这些测试覆盖：生成前gate失败0call/通过1call且原参数不变、binding恢复；context内与退出后input-guard收据SHA相同（避免冻结worker artifacts失效）；有效配置漂移阻止下一stage；stat/历史SHA漂移及SHA超时停止；四模型顺序；重复阈值、并列与partial边界。测试不是GPU生成成功的证明，也不能证明给定240秒一定足够加载当前模型。

| 已审文件 | SHA256 |
|---|---|
| scripts/run_repetition_onset.py | `688955c31abc04448a9b577010f9369ae35e59896dd5d38153e61b87a0b080db` |
| src/asr_lab/repetition_metrics.py | `95f232d3e46b258ce07096d08ef4fcfc8201003635aa49f9ec609496035c1596` |
| tests/test_repetition_onset.py | `eba120fd35f309ab6018525f55f3e5d66745903a146b683e93fec9020b925aaa` |
| reports/repetition-onset/inputs/model-reference.json | `812ec7ed461720455e2cd192cc6ff97e227a1a703c5e2af4e4d9678d8e9e893a` |

资源空闲是实际启动时条件，代码审阅不能预留GPU。即便准入通过，worker前GPU检查仍需生效。报告不得从一次单句对照扩展到总体WER、未保存中间更新或特定训练机制。
