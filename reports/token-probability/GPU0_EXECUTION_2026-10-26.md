# GPU0 A阶段执行与数值验收（2026-10-26）

**模型取证已完成，严格数值验收未通过。** GPU0运行了原始、50步、100步三个模型，共3次生成＋9次显式teacher forward、0训练更新。远端campaign保持真实`complete`；独立CPU数值审计另记`failed_no_retry`，不能把两者合并为“全部通过”。

## 用户授权的Demo暂停与恢复

先按原PID/start_ticks/uid/argv核对PersonaPlex，保存原cwd、解释器、允许列表环境、已保留的`server_prefix_paced`模块SHA及恢复脚本。使用pidfd仅向PID2154793发送SIGTERM，正常退出，无SIGKILL；未向其他进程发信号。CPU网关2184143/8768保留，不删除模型、缓存、环境或迁移保留目录。

停止后第一次GPU0无compute但利用率残留1%，未派发模型；复查0%后仅派发一次A。这个资源复查不是模型重试。暂停/启动收据见[原始授权操作记录](validation/gpu0-pause-launch-20261026.json)与[实际单次launch](validation/gpu0-launch-after-pause-20261026.json)。

[恢复说明](demo-recovery-20261026/README.md)与[恢复脚本](demo-recovery-20261026/restore_personaplex.py)在本地和远端`/mnt/raid/asr-lab/operations/personaplex-pause-20261026/`保留。`--check-only`资产检查已通过，**没有执行实际恢复或语音验收**。需要恢复时，在服务器GPU0空闲且A已退出后手动运行：

```bash
python3 /mnt/raid/asr-lab/operations/personaplex-pause-20261026/restore_personaplex.py
```

A结束后Demo保持暂停，未自动重启。

## 实际范围与资源

- run：`/mnt/raid/asr-lab/runs/qwen-token-probability-20261026-gpu0-a`；固定GPU0 UUID `GPU-eed83181-48a7-e9f3-cb49-e851dfaf647c`。
- 已验证源码提交`6acd9da`，12份部署文件重新匹配SHA；只使用已经验证的GPU0资源身份例外，未改模型、数据、生成或统计源码。
- 启动：01:03:31 CDT；总远端墙钟514.432秒（8.57分钟），包括导入、初始化、SHA、模型、保存、验证、汇总和清理；总上限1200秒。三模型完整stage的保守时间上界160.134/199.967/164.815秒，均小于240秒。
- 原预算3 generate＋9显式TF、512新token、0更新保持。原始run130文件、1,462,218,349bytes（约1.36GiB），其中12份完整词表payload共1,400,262,624bytes；未超过2GiB增量上限。
- 52次预检采样，首条约9.503秒、位于权重SHA前；间隔中位9.998秒/最大10.140秒，终态约514.428秒正常停止。只有采样证据，不把离散显存峰值称绝对峰值。逐worker旧monitor字段含training/checkpoint字样，不表示本轮训练。
- 三worker exit0，cleanup0.241秒，owned进程清空。终态GPU0回到14MiB/0%；旧Early100 SHA未改，其他任务未停止。

[监控与9份TF小收据独立审核](attempts/qwen-token-probability-20261026-gpu0-a/monitor-audit.md)通过：空cache、冻结目标、EOS与`136+j`行对齐正确。该结论不替代数值概率验收。

## 可以确认的原始生成结果

| 权重 | 返回token数（含EOS） | EOS位置（从0计） | 原始轨迹对照 |
|---|---:|---|---|
| 原始模型 | 27 | 26 | 逐token匹配冻结的正确自然大小写轨迹 |
| 50步 | 37 | 36 | 逐token匹配冻结的正确全大写轨迹 |
| 100步 | 512 | 无 | 逐token匹配原CULINARY循环轨迹 |

三个模型的raw tokens、EOS、解析前后文本、完整生成logits/scores、9组TF矩阵和输入行映射均已保存。100步仍复现512窗口内170次CULINARY加末尾CUL；本次没有改变EOS、解码或512上限来消除症状。

## 数值验收失败：保留门槛，不重试

独立审计在服务器CPU运行，CUDA隐藏、4线程，只读保存的矩阵及小型tokenizer，不加载模型权重。沿用原600秒上限、FP64方法与原容差；唯一一次尝试15.597秒后exit1。

首个数据组按固定迭代顺序为`base/generation`；异常栈只直接打印`j=2`。该行`target.prob`与独立FP64复算绝对差 **1.1167530824529592e-6**，超过固定容差 **1e-6**。完整2304行的数值审计没有完成，不能声称全行通过或发布全局误差上界。

[失败报告](attempts/qwen-token-probability-20261026-gpu0-a/analysis/independent-audit.md) · [命令、工具SHA及退出码](attempts/qwen-token-probability-20261026-gpu0-a/analysis/audit-execution.json) · [完整异常栈](attempts/qwen-token-probability-20261026-gpu0-a/analysis/audit-stderr.log)。原始worker FP32摘要只作未验收的派生数据保留；不据其概率高低定稿训练机制结论，未绘制概率结论图。没有放宽容差、修改原始数据或再次执行数值审计。

这是一项派生统计数值验收失败，不能仅据此断言模型加载、原始生成或训练出现新故障；也不能把小误差当理由跳过预定门槛。下一步候选应先用现有矩阵定位CPU统计问题，另立CPU复算版本和验收，不需要重新生成或训练。本轮到此停止，不自动进入B。

## 完整保存与复算边界

原始130文件已全部镜像本地，逐文件SHA与远端库存一致：[完整性收据](attempts/qwen-token-probability-20261026-gpu0-a/artifact-verification.json)。plain rsync先触发300秒传输超时，随后无损压缩续传完成；传输不增加模型调用。核对全部原始文件后仅清理98MB孤立rsync碎片，原始失败与日志均保留。

- 远端raw：`/mnt/raid/asr-lab/runs/qwen-token-probability-20261026-gpu0-a`。
- 本地raw：`reports/token-probability/attempts/qwen-token-probability-20261026-gpu0-a/`，含全部大矩阵与逐行JSON；Git排除PT和大体积逐行派生JSON，但保留小收据、状态、原始token/EOS、SHA清单及报告。
- 模型源每stage做一次完整SHA，之后是stat/软链身份核验，不冒称结束后再次完整权重rehash。
- Demo恢复检查是资产/语法检查，未重新启动模型；旧Early100 failed不变，未补做other/test、训练或B。

最终收尾复查（01:33:38 CDT）：两卡均14MiB/0%，无compute app；本次worker/采样PID均不存在、全局锁已释放。8998未监听，原CPU网关8768仍在，Demo没有自动恢复。[终态主机收据](validation/gpu0-post-a-host-20261026.json)。数值审计确已执行，因早停未输出可靠的全量通过行数，机器状态以null表示该计数，而非把已执行的审计写成0行。
