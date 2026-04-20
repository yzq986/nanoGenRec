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

**优先级**: P0
**来源**: 原创 + SASRec-T (CIKM 2020), TiSASRec (WSDM 2020)
**状态**: 待实现

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

**优先级**: P0
**来源**: 原创 + HSTU1B (KDD 2026), OneLoc Multi-behavior, GR4AD Value-Aware
**状态**: 待实现

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

**优先级**: P0
**来源**: 原创; 类似 BERT 的 segment embedding 设计
**状态**: 待实现

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

**优先级**: P1
**来源**: OneLoc §Category Prompt, UniRec Chain-of-Attribute, GeoGR
**状态**: 待讨论 — 需要品类数据接入

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

**优先级**: P1
**来源**: MTGR Dynamic Masking, HPGR Preference Attention
**状态**: 待讨论

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

**优先级**: P2
**来源**: Time-LLM (ICLR 2024), TiSASRec (WSDM 2020)
**状态**: 待讨论 — IDEA-feat-0 验证后考虑

### 核心思想

IDEA-feat-0 用离散分桶编码时间间隔，丢失了精确时间信息。更高级的方案:

**方案 A — Time-aware RoPE**:
- 用实际 timestamp 而非整数 position 作为 RoPE 的 θ 参数
- `θ_i = ts_i * base^{-2i/d}` (将 timestamp 编码进旋转角度)
- 自然捕获相对时间关系

**方案 B — Continuous Time Kernel**:
- `time_emb = MLP(log(Δt))` → 连续映射，无需分桶
- 精度更高但需要更多参数

### 与 IDEA-feat-0 的关系

- feat-0 (分桶) 是 P0，简单高效，先验证"时间信息是否有用"
- feat-5 (连续) 是 P2，feat-0 验证有效后再升级编码方式

### 关键问题

1. Time-aware RoPE 需要替换当前的 absolute position embedding → 改动较大
2. Continuous time kernel 的 MLP 是否能学好 log-scale 的时间模式
3. 序列内 timestamp 的数值范围: 14天 ≈ 1.2M 秒, 需要归一化

---

## 优先级总结

| 优先级 | ID | 特征 | 收益 | 实现成本 | 建议 |
|--------|-----|------|------|---------|------|
| P0 | IDEA-feat-0 | Time Gap Embedding | ★★★ | 低 | 首选，一次实验验证 side info 价值 |
| P0 | IDEA-feat-1 | Action Type Embedding | ★★☆ | 低 | 和 feat-0 一起做 |
| P0 | IDEA-feat-2 | Segment Embedding 解耦 | ★★ | 极低 | 顺手改，几行代码 |
| P1 | IDEA-feat-3 | Item Category Token | ★★★ | 中 | 需要品类数据，第二轮 |
| P1 | IDEA-feat-4 | User Profile Prefix | ★★ | 中 | 序列够长时不急 |
| P2 | IDEA-feat-5 | Continuous Time Encoding | ★★ | 中 | feat-0 验证后升级 |

**建议实验顺序**: feat-0 + feat-1 + feat-2 三合一 → 一轮实验验证 side information 的整体增量价值。如果有效，再推进 feat-3 (需要品类数据)。

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
