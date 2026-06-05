# Engineering Log

[English](README.md) | [中文](README.zh.md)

Engineering notes for GPU training issues, debugging paths, and fixes.

Each entry records the symptom, root cause, solution, and validation result. The goal is to preserve why a change was made, not only what changed, so similar future issues are easier to diagnose.

## Index

| # | Date | Title | Files |
|---|------|-------|-------|
| 001 | 2026-04-19 | [DPO Gradient Checkpointing + MoE Compatibility + Memory Fragmentation](./001-dpo-oom-gradient-checkpointing.md) | `rl/dpo.py`, `rl/trainer.py`, `ntp/model.py` |
