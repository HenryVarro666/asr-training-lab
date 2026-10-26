# PersonaPlex Demo暂停与恢复

用户2026-10-26明确授权暂停Demo后端，释放GPU0执行有界ASR A。只停止已核对身份的PID2154793；保留CPU网关8768/PID2184143及其他任务。不删除模型、虚拟环境、迁移保留副本或日志。

远端恢复包：`/mnt/raid/asr-lab/operations/personaplex-pause-20261026/`。

检查恢复资产（不启动模型）：

```bash
python3 /mnt/raid/asr-lab/operations/personaplex-pause-20261026/restore_personaplex.py --check-only
```

需要恢复Demo时，确认A已退出且GPU0空闲，再手动执行：

```bash
python3 /mnt/raid/asr-lab/operations/personaplex-pause-20261026/restore_personaplex.py
```

脚本核对原模块SHA、缓存路径、GPU0及8998端口，按保存的cwd/argv/允许列表环境重启原`server_prefix_paced`，保存新PID与日志。它不会准备/替换模型或启动实验8999，不会重启网关。不要source通用RAID env.sh以免改变缓存位置。随后检查日志、8998监听及8768状态；启动派发不是语音功能验收。A结束后不会自动重启Demo。
