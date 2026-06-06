# Experiment Logs

[English](README.md) | [中文](README.zh.md)

按研究阶段组织的人类可读实验记录。

此目录用于结论、比较和下一步规划。代码接口和实现细节属于模块 README，原始产物保存在 `experiments/` 下。

## 阶段总结

| 阶段 | 范围 | 当前最佳 / 状态 | 摘要 |
|------|------|-----------------|------|
| [tokenizer/](tokenizer/README.md) | EXP-001 至 EXP-012, EXP-026, EXP-045, EXP-049 | 推荐 SID 缓存：`exp049-0.6b-nc8192-h128`, `exp049-4b-nc8192-h128` | Semantic ID 码本、碰撞、基尼系数和 snHR。 |
| [ntp/](ntp/README.md) | EXP-013 至 EXP-016, EXP-036, EXP-041 至 EXP-050 | M-tier R@500=70.2%; L-tier SFT R@500=64.1% | 自回归推荐器缩放和特征消融。 |
| [rl/](rl/README.md) | EXP-017 至 EXP-040 | ECPO R@500=65.7%（S-tier 流水线） | 偏好数据、DPO、GRPO 和 ECPO 对齐。 |

## 实验条目格式

每个 `exp-NNN.md` 应无需打开训练日志即可阅读：

```markdown
## EXP-NNN: 简短标题

**日期**: YYYY-MM-DD
**状态**: completed

### 背景

此次运行试图回答什么问题？

### 设计

- 变量：
- 固定：
- 基线：

### 结果

| 配置 | R@10 | R@500 | PPL |
|------|------|-------|-----|

### 分析

什么变了，什么没变，为什么？

### 下一步

接下来应运行或更改什么？
```

## 维护规则

- 每当完成的实验更改了 SOTA 表、基线或建议时，更新阶段 README。
- 仅在对新读者重要的标题级更改时更新根 README。
- 明确标记无效或有 bug 的实验；不要从实验谱系中删除。
- 使用全量评估数字进行比较。训练中的 inline 评估是健康检查，不是可发布的基线。

