# Truncation probe 独立 CPU 审计

**结论：通过本次证据完整性与分层一致性审计；原 Early100 campaign 仍为 failed。**

核验全部 3 次 probe、33 条输入：24 个 worker artifacts 的 SHA256/大小；PT 与 JSON 张量逐值、prompt 前缀及 continuation 切分；实际 EOS/PAD；远端隐藏 CUDA 的离线 tokenizer/token pieces/两种 decode/官方 parser；本地独立 normalizer 与 S/D/I/H。全部一致。完整命令、Python 代码、版本、源码与 57 个来源指纹见 [independent-audit.json](independent-audit.json)。

| 实际阶段 | 条数 | prompt padded width | 目标 continuation / EOS | 实际解析前文本 | 解析结果 |
|---|---:|---:|---|---|---|
| batch16-first | 16 | 200 | 512 / 无 EOS | 170 × CULINARY + CUL | CULINARY CUL |
| batch16-repeat | 16 | 200 | 512 / 无 EOS | 170 × CULINARY + CUL | CULINARY CUL |
| target-single | 1 | 137 | 512 / 无 EOS | 170 × CULINARY + CUL | CULINARY CUL |

上述重复来自三次实际保存的 token/解码，不是 synthetic 测试推断。目标保留/跳过特殊 token 的解码相同；官方重复修复将实际长重复串压缩为两词。目标解析后 S/D/I/H = 2/14/0/0，参考词数16；基线同句 0/0/0/16。

两次 batch 的全部输入张量、effective config、完整 tokens 与解析结果相同；单句目标 continuation 也相同。三次运行合计 130.6929 秒，3 次 generate、33 次样本消费、0 optimizer updates。

监控共10个离散样本未记录 alert；不能排除采样间的瞬时峰值。已有最终主机收据记录 GPU1 近空闲、GPU0 PersonaPlex 保留、全局锁可用；这里未重新进行 live 主机检查。

解释边界：原 Early100 raw tokens 未保存，前137批未重放；本次复现不能证明旧运行的完整 token 逐字一致。没有收集 logits、确认训练/数值根因或进行人工声学验证；没有修改 checkpoint、512-token 上限、EOS、prompt 或原质量门槛。原 failed campaign 与原 reference 来源 SHA 保持不变。
