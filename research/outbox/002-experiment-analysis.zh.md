---

[English](002-experiment-analysis.md) | [中文](002-experiment-analysis.zh.md)
date: "2026-04-22 18:35"
type: finding
priority: normal
subject: "EXP-022~025 趋势分析 + 下一步实验提案"
needs_response: true
---

## 一、实验结果总览

完成了对 EXP-022 至 EXP-025 全部结果的收集和分析。

### 当前完整 leaderboard (R@500 排序)

| Rank | Config | PPL | R@10 | R@500 | 关键变量 |
|------|--------|-----|------|-------|---------|
| **1** | **exp025-beam-passes** | **25.22** | 10.4% | **63.6%** | seg+time+action, beam传features |
| 2 | exp023-segment | 25.94 | 10.9% | 61.2% | 仅segment_emb |
| 3 | exp023-baseline | 28.41 | 11.0% | 60.7% | EXP-016复现 |
| 4 | exp023-timegap | 28.78 | 10.9% | 60.1% | 仅time_gap (有泄漏) |
| 5 | exp024-seg-timegap | ~26 | — | 59.8% | seg + shifted time_gap |
| 6 | exp022-alpha001 | 27.89 | 10.3% | 59.2% | contrastive α=0.01 |
| 7 | exp022-dim256 | 29.66 | 10.1% | 58.8% | contrastive dim=256 |
| 8 | EXP-016 baseline | 27.05 | 9.9% | 58.5% | 原始baseline |
| 9 | exp022-temp005 | 28.16 | 10.1% | 58.2% | contrastive τ=0.05 |
| 10 | exp022-alpha01 | 29.22 | 9.7% | 57.9% | contrastive α=0.1 |
| 11 | exp022-alpha05 | 29.04 | 9.7% | 56.3% | contrastive α=0.5 |
| 12 | exp024-seg-all | ~26 | — | ~55% | seg + shifted all |
| 13 | exp024-seg-action | ~27 | — | 52.9% | seg + shifted action |
| 14 | exp023-all | 25.16 | 9.5% | 55.0% | all features (有泄漏) |
| 15 | exp025-action-l2only | 24.85 | 5.5% | 27.0% | action仅L2 (失败) |
| 16 | exp023-action | 27.50 | 4.9% | 28.5% | 仅action (严重泄漏) |

## 二、关键发现

### 1. Beam search feature passing 是正确方向 ✅
EXP-024 (shift) 失败后，EXP-025 beam_passes 彻底解决了 time_gap/action 的训练-推理 gap：
- R@500: 61.2% → **63.6%** (+2.4pp)，PPL: 25.94 → **25.22** (-0.72)
- 方法：训练正常使用所有features，beam search时传入 time_gap(已知真值) + action(carry-forward上一个context item)
- 这证明 side features 本身有价值，之前失败是推理端bug，不是特征无用

### 2. Contrastive loss 无效 ❌
EXP-022 全5个config都不如baseline。IDEA-onemall-0 可以 close。
- 最好的 α=0.01 也只+0.7pp R@500，代价是 +0.84 PPL
- α 越大越差（0.1 → -0.6pp, 0.5 → -2.2pp）
- 根因推测：SID 是离散 codebook token，embedding 空间对齐对离散 token 预测无直接帮助

### 3. Shift 方案完全错误 ❌
EXP-024 证明 shift 不仅没修复 gap，还让信号变陈旧导致更差。正确的设计方向是修复推理端而非修改训练数据。

### 4. Segment embedding 稳定正向 ✅
EXP-023 → EXP-025，segment_emb 始终是正向信号，且与 beam_passes 正交叠加。

## 三、P0 Ideas 状态

| IDEA | 状态 | 结论 |
|------|------|------|
| IDEA-feat-0 (time_gap) | ✅ 已验证 via EXP-023/025 | 有效，需配合beam_passes |
| IDEA-feat-1 (action_level) | ✅ 已验证 via EXP-023/025 | 有效，需配合beam_passes |
| IDEA-feat-2 (segment_emb) | ✅ 已验证 via EXP-023 | 确认正向 |
| IDEA-onemall-0 (contrastive) | ❌ 已测试 via EXP-022 | 无效，discard |
| IDEA-genrec-0 (PW-NTP) | 🔲 未测试 | **最高优先级下一步** |

## 四、下一步实验提案

### 提案 A: EXP-026 — Page-wise NTP (IDEA-genrec-0) ⭐ 推荐

**为什么现在做这个**：
- 已读 GenRec 论文 (2604.14878)，JD在线 +9.5% click，幻觉率降50%
- 与 beam_passes 完全正交 — PW-NTP 改训练目标，beam_passes 改推理端
- **推理端完全不需要改动**（训练-推理不对称是有意设计）
- 需要修改的模块：`ntp/preprocess.py`（数据构造）、`ntp/train.py`（dataloader）
- 不需要修改 `ntp/model.py`（模型结构不变）

**设计草案**：
- 数据构造：同一 session/时间窗口内多个正交互 item 拼接为 target 序列
- Baseline: exp025-beam-passes (R@500=63.6%)
- 对比：point-wise NTP vs page-wise NTP (2/3/5 items per page)
- 预计训练时间与当前相同（packed sequences，总 token 数不变）

**需要人类确认**：
1. 这涉及修改 `ntp/preprocess.py` 和 `ntp/train.py` 两个源码文件 — 是否授权？
2. 数据中 session 分割的标准？按时间间隔 >30min 切分？还是按天？

### 提案 B: EXP-026 — 在 beam_passes 基础上做 scale-up 验证

**为什么考虑**：
- beam_passes 是新 best，但只在 S-tier (17.5M) 上验证
- 可以尝试 M-tier 或 L-tier 模型看 side features 的收益是否随模型变大而增大
- 不需要源码修改，纯 config 调整

**缺点**：缺少新idea，更像是 validation 而非 exploration

---

**我的推荐**：优先做 **提案 A (Page-wise NTP)**。理由：
1. 它是目前未测试的最高影响力 idea（论文报告 HR@50: 0.62→0.72，巨大提升）
2. 与已有 beam_passes 改进正交，可以叠加
3. 改动范围明确（preprocess + dataloader），不涉及模型结构
4. 需要源码修改，所以需要人类授权才能进行 → 尽早提出

请回复是否同意提案 A，以及 session 分割标准的偏好。
