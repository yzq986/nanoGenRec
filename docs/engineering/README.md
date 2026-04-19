# Engineering Log

GPU 训练过程中遇到的工程问题、调试过程和解决方案的记录。

每条记录包含：问题现象、根因分析、解决方案、验证结果。
重点记录**为什么**这样改，而不只是改了什么，方便未来遇到类似问题时快速定位。

## Index

| # | Date | Title | Files |
|---|------|-------|-------|
| 001 | 2026-04-19 | [DPO Gradient Checkpointing + MoE 兼容 + 显存碎片化](./001-dpo-oom-gradient-checkpointing.md) | `rl/dpo.py`, `rl/trainer.py`, `ntp/model.py` |
