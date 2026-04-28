# NTP Side Information & Feature Engineering

当前 NTP 模型输入**仅有 SID token 序列 + positional embedding**，无任何 side information。本文件收集所有"给 NTP 注入额外特征"的优化方向。

**影响范围**: `ntp/preprocess.py`, `ntp/train.py`, `ntp/model.py`

---

## 演进路径

```
纯 SID token + position embedding (当前 baseline, EXP-016)
├── IDEA-feat-0: Time Gap Embedding (相邻 item 时间间隔)
│   └── 区分"连续刷" vs "隔天回来"，学习兴趣衰减
├── IDEA-feat-1: Action Type Embedding (行为强度信号)
│   └── 区分 click/like/fav/share，强交互 > 弱交互
├── IDEA-feat-2: Segment Embedding 解耦 (item_pos + layer_pos)
│   └── 当前 position 同时编码两个维度，解耦后各自独立学
├── IDEA-feat-3: Item Category Token (品类 side information)
│   └── 帮助 cold-start + 跨品类泛化
├── IDEA-feat-4: User Profile Prefix Token (全局用户画像)
│   └── 短序列冷启动先验
└── IDEA-feat-5: Relative Time Encoding (连续时间 → RoPE 式编码)
    └── 比 bucketed gap 更精细的时间建模
```

---

## IDEA-feat-0: Time Gap Embedding (时间间隔特征)

**优先级**: ~~P0~~ → ✅ 已验证有效
**来源**: 原创 + SASRec-T (CIKM 2020), TiSASRec (WSDM 2020)
**状态**: ✅ 已实现并验证 — EXP-036 clean ablation: +3.7pp R@500 (B vs A)

### 核心思想

当前 NTP 只用 chronological 顺序，丢失了时间密度信息。相邻 item 间的时间间隔 `Δt = ts[i] - ts[i-1]` 包含关键信号：
- Δt < 1 min → 连续刷，同 session 内兴趣延续
- Δt ~ 30 min → session 切换
- Δt > 1 day → 兴趣可能已衰减/转移

### 实现方案

**Preprocess 端** (`ntp/preprocess.py`):
- 计算相邻 item 的 Δt (秒)
- 分桶: 对数分桶 [0, 1min, 5min, 30min, 1h, 6h, 1d, 3d, 7d, 14d+] → ~10-16 个桶
- 每个 item 的 3 个 SID token 共享同一个 time_gap bucket
- 序列第一个 item 的 gap 设为特殊 token (BOS)

**Model 端** (`ntp/model.py`):
```python
self.time_gap_emb = nn.Embedding(n_time_buckets, embed_dim)

# forward 中:
x = token_emb + pos_emb + time_gap_emb[time_gaps]
```

### 改动文件

1. `ntp/preprocess.py` — `save_shard()` 增加 `time_gaps` 数组
2. `ntp/train.py` — `build_unified_sequences()` 计算 Δt 分桶, `NTPDataset.__getitem__()` 返回 time_gaps
3. `ntp/model.py` — `NTPModel.__init__()` 加 `time_gap_emb`, `forward()` 加上

### 评估

- 基线: EXP-016 14d-S (R@500=58.5%)
- 指标: Recall@{10,50,100,500}, 特别关注长间隔 item 的预测准确率
- 消融: 不同分桶粒度 (8 vs 16 vs 32 buckets)

### 关键问题

1. 行为数据中 `first_ts` 精度是秒级还是天级？需确认
2. 分桶边界的设计: 对数 vs 线性 vs learned boundaries
3. 同一 item 的 3 个 SID token 共享同一 time_gap，还是只在 s0 位置注入

---

## IDEA-feat-1: Action Type Embedding (行为类型特征)

**优先级**: ~~P0~~ → ✅ 已验证有效
**来源**: 原创 + HSTU1B (KDD 2026), OneLoc Multi-behavior, GR4AD Value-Aware
**状态**: ✅ 已实现并验证 — EXP-036 包含 `--use_action_level`，Config B vs A: +3.7pp R@500

### 核心思想

当前 `action_bitmap > 0` 只做正负过滤，丢弃了行为强度信息。bitmap 的不同 bit 代表不同行为（click, like, fav, share, comment, purchase），这些行为表达了不同的兴趣强度。

### 实现方案

**方案 A — Action Level (离散强度)**:
```python
# 将 action_bitmap 映射为 action_level:
# click=1, like=2, fav=3, share=4, purchase=5
action_level = compute_action_level(action_bitmap)
self.action_emb = nn.Embedding(n_action_levels, embed_dim)
```
- 加到每个 item 的 s0 位置 (或 3 个 token 都加)

**方案 B — Multi-hot Action Embedding**:
```python
# 保留 bitmap 的 multi-hot 性质:
self.action_proj = nn.Linear(n_action_bits, embed_dim)
action_vec = bitmap_to_multihot(action_bitmap)  # (n_bits,)
action_emb = self.action_proj(action_vec)
```

### 与 IDEA-oneloc-5 (Multi-behavior) 的关系

- oneloc-5 是把不同行为的序列**分离**（click seq, buy seq, ...）
- 本 IDEA 是在**统一序列中标注行为类型**，保持时间顺序
- 两者正交：本 IDEA 更轻量，oneloc-5 更彻底

### 改动文件

1. `ntp/preprocess.py` — `save_shard()` 增加 `action_levels` 数组
2. `ntp/train.py` — `_build_user_items()` 保留 action_bitmap, 计算 action_level
3. `ntp/model.py` — `NTPModel` 加 `action_emb`

### 评估

- 对比: 无 action_emb vs action_level vs multi-hot
- 分析: 高价值行为 (purchase/fav) 的 item 预测准确率是否提升

### 关键问题

1. 当前数据中 action_bitmap 的各 bit 含义需要确认 (grep `_VIEW_EXIT_BIT`)
2. 行为分布是否严重不均衡 (click >> purchase)
3. 模型是否会过拟合 action type 而忽略序列模式

---

## IDEA-feat-2: Segment Embedding 解耦 (Item Position + Layer Position)

**优先级**: ~~P0~~ → ✅ 已验证有效
**来源**: 原创; 类似 BERT 的 segment embedding 设计
**状态**: ✅ 已实现并验证 — EXP-036 包含 `--use_segment_emb`，Config B vs A: +3.7pp R@500

### 核心思想

当前 `pos_emb[i]` 同时编码了两个信息:
- "这是第几个 item" (item_pos = i // n_sid_layers)
- "这是 item 内第几层 SID" (layer_pos = i % n_sid_layers)

两个维度耦合在同一个 embedding 中，模型需要自己学会 disentangle。解耦后各维度独立学习:

```python
self.item_pos_emb = nn.Embedding(max_items, embed_dim)
self.layer_pos_emb = nn.Embedding(n_sid_layers, embed_dim)  # 只有 3 个

# forward:
item_pos = positions // n_sid_layers
layer_pos = positions % n_sid_layers
x = token_emb + item_pos_emb[item_pos] + layer_pos_emb[layer_pos]
```

### 收益

- `item_pos_emb` 可以学到纯粹的"序列位置→兴趣衰减"信号
- `layer_pos_emb` 可以学到"L0 是粗粒度类目, L1 是细粒度, L2 是消歧"的结构
- 参数量更少: `max_items + 3` vs `max_items * 3`

### 改动文件

1. `ntp/model.py` — 替换 `self.pos_emb` 为 `self.item_pos_emb` + `self.layer_pos_emb`

### 评估

- 直接替换, 对比 loss 和 Recall@K
- 可视化: item_pos_emb 是否呈现衰减趋势, layer_pos_emb 是否 L0/L1/L2 明显不同

### 关键问题

1. 当前 pos_emb 参数量 = max_seq_len * embed_dim; 解耦后 = (max_items + 3) * embed_dim，更少
2. 解耦是否会损失 position-layer 交互信息？可用 `item_pos_emb + layer_pos_emb + interaction_emb` 保留

---

## IDEA-feat-3: Item Category / Attribute Token

**优先级**: ~~P1~~ → P4 暂缓
**来源**: OneLoc §Category Prompt, UniRec Chain-of-Attribute, GeoGR
**状态**: 暂缓 — 品类信息已通过 Qwen3 text embedding 隐式编码；EXP-036 full-features (time_gap+action_level+segment) 已达 R@500=59.0%，品类 token 的边际增益预期有限。需要先完成 EXP-037→039 RL 链路再评估

### 核心思想

SID 是语义压缩 (embedding → 离散码)，但丢失了品类/作者等结构化信号。补充 category token 可以帮助模型理解跨品类泛化和 cold-start item。

### 方案对比

**方案 A — Category Token Prefix** (改序列结构):
```
序列: [cat_1, s0_1, s1_1, s2_1, cat_2, s0_2, s1_2, s2_2, ...]
```
- 每个 item 前插一个 category token
- 序列长度 +33%
- 模型可以从 cat token 预测到 s0 (coarse→fine)

**方案 B — Category Embedding 加法** (不改序列长度):
```python
self.cat_emb = nn.Embedding(n_categories, embed_dim)
# 只在 s0 位置 (或全部 3 个位置) 加上 category embedding
x[s0_positions] += cat_emb[category_ids]
```
- 序列长度不变
- 但 category 信息可能被 token embedding 淹没

**方案 C — Category 作为 Soft Prompt** (参考 IDEA-glide-0):
```
序列: [soft_prompt(category), s0_1, s1_1, s2_1, ...]
```
- 用 learned soft prompt 编码 category, 作为序列前缀

### 与 IDEA-unirec-0 (Chain-of-Attribute) 的关系

- UniRec 让模型**依次生成** category → brand → SID (chain-of-thought 式)
- 本 IDEA 只是把 category 作为**输入特征**注入, 不改变生成目标
- UniRec 更彻底但改动更大 (需要改生成目标和解码策略)

### 改动文件

1. 新增数据: item → category 映射表
2. `ntp/preprocess.py` — 加载 category 映射, 生成 category 序列
3. `ntp/model.py` — 加 `cat_emb`, forward 中注入

### 关键问题

1. **数据可用性**: 当前数据中是否有品类信息？需要从原始 item metadata 获取
2. 品类粒度: 一级品类 (~20) vs 二级品类 (~200) vs 三级品类 (~2000)
3. 方案 A 增加序列长度→增加训练成本; 方案 B 零成本但效果可能弱

---

## IDEA-feat-4: User Profile Prefix Token (用户画像 token)

**优先级**: ~~P1~~ → P4 暂缓
**来源**: MTGR Dynamic Masking, HPGR Preference Attention
**状态**: 暂缓 — Align³GR ablation 显示用户特征 +0.9pp，收益有限；短序列冷启动场景先验价值有限，NTP 序列本身已隐式建模用户偏好。RL 对齐链路 (EXP-037→039) 优先级更高

### 核心思想

用用户历史统计构造一个 "user summary" token 放在序列最前面，给模型提供全局先验:
- Top-3 L0 频率分布 (主要兴趣类目)
- 活跃度 bucket (日活/周活/月活)
- 平均 action 强度 (高互动用户 vs 浏览型用户)

### 实现方案

```python
# Preprocess: 计算用户级统计
user_profile = {
    'top_l0': [most_frequent_l0_clusters],  # top-3
    'activity_bucket': activity_level,       # 0-7
    'avg_action_level': mean_action_level,   # 0-4
}

# Model:
self.profile_emb = nn.Sequential(
    nn.Linear(profile_dim, embed_dim),
    nn.GELU(),
    nn.Linear(embed_dim, embed_dim)
)

# Forward: 作为序列第一个 token
profile_token = self.profile_emb(user_profile_features)
x = torch.cat([profile_token.unsqueeze(1), token_sequence], dim=1)
```

### 收益

- 短序列用户: profile token 提供先验，弥补历史不足
- 条件化生成: 模型可以根据 user type 调整预测分布
- 与 IDEA-sigma-0 (指令多任务) 的 instruction prefix 方向一致

### 关键问题

1. Profile 计算需要基于训练窗口**之前**的数据，避免信息泄露
2. 如果序列已经够长 (>50 items)，profile token 的边际收益可能很小
3. 与 IDEA-feat-0 (time gap) 和 IDEA-feat-1 (action type) 有部分信息重叠

---

## IDEA-feat-5: Relative / Continuous Time Encoding

**优先级**: P2 → **升级为 P1**
**来源**: Time-LLM (ICLR 2024), TiSASRec (WSDM 2020), **TO-RoPE (arxiv 2510.20455, Roblox)**, **SynerGen (arxiv 2509.21777)**
**状态**: feat-0 已验证有效 (EXP-025 R@500=63.6%)，准备进入 TO-RoPE 实验

### 核心思想

IDEA-feat-0 用离散分桶编码时间间隔，丢失了精确时间信息。更高级的方案:

**方案 A — Time-and-Order RoPE (TO-RoPE)** [新增 — 强烈推荐]:
TO-RoPE 将 RoPE 扩展为同时编码 discrete index 和 wall-clock time：
- `θ_k(i) = (1-λ_k) · α_p · i · ω_p_k + λ_k · α_t · τ_i · ω_t_k`
- 三种实例化:
  - **Split-by-dim** (推荐): 一部分 rotary plane 用 index，另一部分用 time。无 plane 内干扰，显式容量分配 (如 70% time / 30% index)
  - **Split-by-head**: 部分 head 用 index-only，部分 head 用 time-only。head 间通过 output projection 融合
  - **Early fusion**: index 和 time angle 直接相加。最灵活但有破坏性干扰风险
- Roblox 大规模工业数据: split-by-dim/head 一致优于 APE、HSTU-style relative bias、index-only RoPE
- 最佳 split ratio: 0.3-0.5 (time 占 30-50% capacity)
- 天然捕获三种时间模式: short-term burstiness, long-term rhythms, temporal periodicity (day-of-week/time-of-day)
- 兼容 FlashAttention，几乎无额外参数

**方案 B — Continuous Time Kernel**:
- `time_emb = MLP(log(Δt))` → 连续映射，无需分桶
- 精度更高但需要更多参数

**方案 C — BOS 时间注入 (OneLive)**:
- OneLive (快手, arxiv 2602.08612): 在 [BOS] token 注入 multi-granular temporal features
- `x_BOS = x_BOS + MLP(Concat(x_Hour, x_Day, x_Week))`
- 只在序列开头注入全局时间，不改位置编码
- 轻量，但只有 "当前时刻" 的全局信号，不编码序列内各 item 的时间关系

### 与 IDEA-feat-0 的关系

- feat-0 (分桶) 是 P0，已验证有效 — EXP-025 beam passes: R@500 61.2% → 63.6%
- feat-5 (TO-RoPE) 是 P1，**替换 absolute pos_emb + time_gap_emb 为统一的 TO-RoPE**
- 关键优势: TO-RoPE 同时编码 order + time，不需要单独的 time_gap 分桶特征

### 对我们的适配

当前模型用 absolute learnable position embedding + time_gap bucket embedding (additive)。切换到 TO-RoPE:
1. 移除 `pos_emb` 和 `time_gap_emb`
2. 每个 item 关联一个 normalized timestamp `τ_i = (ts_i - ts_0) / scale`
3. 同一 item 的 3 个 SID token 共享同一 `τ_i`，但 index 是 `3*item_idx + layer`
4. Split-by-dim: 例如 256d → 128d for index planes + 128d for time planes
5. Beam search 生成时: index 自然递增，τ 用 target item 的真实 timestamp

**注意**: 这解决了 EXP-023/024/025 中的核心问题 — time_gap_emb 作为 additive feature 在 beam search incremental path 需要手动传递。TO-RoPE 把时间信息编码到 Q/K 旋转中，KV cache 自然保留时间信息。

### 关键问题

1. ~~Time-aware RoPE 需要替换当前的 absolute position embedding → 改动较大~~ TO-RoPE 论文已验证可行性
2. Continuous time kernel 的 MLP 是否能学好 log-scale 的时间模式
3. 序列内 timestamp 的数值范围: 14天 ≈ 1.2M 秒, 需要归一化 → TO-RoPE 建议用 days 或 hours 作为 τ 单位
4. 当前模型用 `nn.MultiheadAttention`，需要确认能否方便地注入 custom RoPE（可能需切换到手写 attention 或 xformers）

---

## 优先级总结

| 优先级 | ID | 特征 | 收益 | 实现成本 | 建议 |
|--------|-----|------|------|---------|------|
| ~~P0~~ → ✅ | IDEA-feat-0 | Time Gap Embedding | ★★★ | 低 | ✅ EXP-036 验证有效，+3.7pp R@500 |
| ~~P0~~ → ✅ | IDEA-feat-1 | Action Type Embedding | ★★☆ | 低 | ✅ EXP-036 包含，三合一联合验证 |
| ~~P0~~ → ✅ | IDEA-feat-2 | Segment Embedding 解耦 | ★★ | 极低 | ✅ EXP-036 包含，三合一联合验证 |
| P4 暂缓 | IDEA-feat-3 | Item Category Token | ★★ | 中 | 品类信息已隐含在 text embedding，边际收益有限 |
| P4 暂缓 | IDEA-feat-4 | User Profile Prefix | ★☆ | 中 | Align³GR ablation 仅 +0.9pp，先做 RL 链路 |
| **P1** | **IDEA-feat-5** | **TO-RoPE Time+Order Encoding** | **★★★** | **中** | **feat-0/1/2 已验证，下一步升级到 TO-RoPE** |

**当前状态**: feat-0/1/2 三合一已通过 EXP-036 验证有效（+3.7pp R@500）。下一步: 完成 RL 对齐链路 (EXP-037→039)，之后考虑 feat-5 TO-RoPE 进一步提升。

---

## 与其他 ideas/ 的关系

| 本文件 | 关联 | 关系说明 |
|--------|------|---------|
| IDEA-feat-1 | IDEA-oneloc-5 (training.md) | oneloc-5 分离行为序列, feat-1 在统一序列中标注行为 |
| IDEA-feat-1 | IDEA-gr4ad-2 (training.md) | gr4ad-2 按行为加权 loss, feat-1 让模型学行为表示 |
| IDEA-feat-3 | IDEA-unirec-0 (architecture.md) | unirec-0 生成 category, feat-3 只注入 category |
| IDEA-feat-3 | IDEA-glide-0 (architecture.md) | glide-0 soft prompt, feat-3 方案 C 类似 |
| IDEA-feat-4 | IDEA-mtgr-0 (training.md) | mtgr-0 dynamic masking 支持 profile 双向 attn |
| IDEA-feat-5 | IDEA-gems-0 (architecture.md) | gems-0 multi-stream temporal, feat-5 是轻量替代 |
