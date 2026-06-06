# Embedding & Semantic ID 评估指标体系

[English](METRICS.md) | [中文](METRICS.zh.md)

> 参考: OneRec (arxiv 2506.13695)

本模块提供两类共 **12 个指标**，用于评估 embedding 模型质量和 RKMeans 量化后的 Semantic ID 质量。

---

## 目录

- [一、Intrinsic Metrics（内在指标，无需行为数据）](#一intrinsic-metrics内在指标无需行为数据)
  - [1. Reconstruction Loss](#1-reconstruction-loss)
  - [2. Codebook Utilization](#2-codebook-utilization)
  - [3. Token Entropy](#3-token-entropy)
  - [4. Cosine Similarity Distribution](#4-cosine-similarity-distribution)
  - [5. Effective Dimension](#5-effective-dimension)
  - [6. Semantic ID Collision](#6-semantic-id-collision)
  - [7. Cluster Balance](#7-cluster-balance)
- [二、Behavior Metrics（行为指标，需要用户行为数据）](#二behavior-metrics行为指标需要用户行为数据)
  - [8. User Semantic Consistency](#8-user-semantic-consistency)
  - [9. Semantic Neighbor Hit Rate](#9-semantic-neighbor-hit-rate)
  - [10. Embedding-Behavior Correlation](#10-embedding-behavior-correlation)
  - [11. Positive-Negative Separation](#11-positive-negative-separation)
  - [12. Semantic ID Prediction (NTP)](#12-semantic-id-prediction-ntp)

---

## 一、Intrinsic Metrics（内在指标，无需行为数据）

### 1. Reconstruction Loss

**目标**: 衡量 RKMeans 量化的精度，即原始 embedding 被量化后能还原多少。

**计算方法**（与 `rkmeans_stage2_train_v2.py` 的 `generate_semantic_ids` 一致）:

```
对于 L 层 RKMeans:
  # Step 0: 仅对原始输入做一次 L2 归一化 (layer 0 only)
  if normalize_residuals:
    residual = L2_normalize(embedding)       # ||residual|| = 1
  else:
    residual = embedding

  input_for_loss = residual                   # 保存归一化后的输入用于计算 loss

  # Step 1..L: 逐层量化，残差不再重新归一化
  for layer_i in [1..L]:
    assignment = argmin ||residual - centroid||^2
    reconstruction_i = centroid[assignment]    # 直接使用 centroid，不做缩放
    residual = residual - reconstruction_i    # 残差保留原始 scale

  x_hat = sum(reconstruction_1, ..., reconstruction_L)
  total_loss = mean(||input_for_loss - x_hat||^2)
  normalized_loss = total_loss / mean(||input_for_loss||^2)
```

> **注意**: `normalize_residuals=True` 仅在 layer 0 对输入做一次 L2 归一化，后续层的残差**不再**重新归一化。
> 这与训练和推理（`generate_semantic_ids`）的逻辑完全一致。

**主指标**: `normalized_loss`（值域 [0, +inf)，越小越好）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `normalize_residuals` | `True` | 仅对 layer 0 输入做 L2 归一化（后续层不再归一化） |
| `chunk_size` | `50000` | 分批处理大小 |

**质量阈值**:

| 等级 | 阈值 |
|------|------|
| Excellent | <= 0.05 |
| Good | <= 0.10 |
| Acceptable | <= 0.20 |
| Poor | > 0.20 |

**举例**: 假设 embedding 维度 D=768，10 万个样本，3 层 RKMeans（每层 256 个 centroid）
- normalized_loss = 0.03 --> Excellent: 仅损失 3% 信息
- normalized_loss = 0.15 --> Acceptable: 损失 15%，建议增加层数或 cluster 数

---

### 2. Codebook Utilization

**目标**: 衡量各前缀深度下 SID 空间的利用率。

**计算方法**:

```
# 逐前缀深度统计 (3 层为例)
depth=1: unique("a_*_*") / N^1     # L1 前缀利用率
depth=2: unique("a_b_*") / N^2     # L1+L2 前缀利用率
depth=3: unique("a_b_c") / N^3     # 完整 SID 利用率

layer_values = [util_depth1, util_depth2, util_depth3]
```

**主指标**: `space_utilization = n_unique_full_sids / N^L`（值域 [0, 1]）

**layer_values**: 各 depth 的 `n_unique_prefix / N^depth`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | `50000` | 分批处理大小 |

**质量阈值**: 无固定阈值（取决于 n_items / N^L 比例），状态为 `unknown`

**举例**: 160K 内容，3 层 1024 clusters
- depth=1: 1020 unique / 1024 = 0.9961 (99.6% L1 cluster 被使用)
- depth=2: 150,000 unique / 1,048,576 = 0.1431 (14.3% L1+L2 组合被使用)
- depth=3: 158,000 unique / 1,073,741,824 = 0.0001 (完整 SID 空间极大)
- 越浅层利用率越高，越深层因组合爆炸而降低，属正常现象

---

### 3. Token Entropy

**目标**: 衡量 SID 分布的均匀程度，按前缀深度分层统计。

**计算方法**:

```
# 主指标: 完整 SID 的 normalized entropy
sid_counts = Counter(all_semantic_ids)
H_full = -sum( p(sid) * log2(p(sid)) )
normalized = H_full / log2(N_total)

# 逐前缀深度 entropy (3 层为例)
depth=1: prefixes = ["a_*_*"]
  H_1 = -sum( p(prefix) * log2(p(prefix)) )
  normalized_1 = H_1 / log2(N)           # 最大熵 = log2(N^1)

depth=2: prefixes = ["a_b_*"]
  H_2 = -sum( p(prefix) * log2(p(prefix)) )
  normalized_2 = H_2 / log2(N^2)         # 最大熵 = log2(N^2)

depth=3: prefixes = ["a_b_c"] (= full SID)
  H_3 = H_full
  normalized_3 = H_3 / log2(N^3)         # 最大熵 = log2(N^3)
```

**主指标**: `normalized_entropy = H_full / log2(N_total)`（值域 [0, 1]，越高越好）

**layer_values**: 各 depth 的 `H_depth / log2(N^depth)`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | `50000` | 分批处理大小 |

**质量阈值**:

| 等级 | 阈值 |
|------|------|
| Excellent | >= 0.95 |
| Good | >= 0.90 |
| Acceptable | >= 0.80 |
| Poor | < 0.80 |

**举例**: 160K 内容，3 层 1024 clusters
- depth=1: H=9.95 / log2(1024)=10.0 → normalized=0.995 (L1 前缀非常均匀)
- depth=2: H=16.8 / log2(1024^2)=20.0 → normalized=0.840 (L1+L2 组合有些集中)
- depth=3: H=17.1 / log2(1024^3)=30.0 → normalized=0.570 (完整 SID 空间极大，集中度更高)
- 主指标 normalized = 17.1 / log2(160000) = 0.989

---

### 4. Cosine Similarity Distribution

**目标**: 检查 embedding 的判别能力——好的 embedding 应该 mean 适中、std 大（有区分度）。

**计算方法**:

```
# 采样 sample_size 个 embedding
sample_norm = L2_normalize(sample)
sim_matrix = sample_norm @ sample_norm.T    # (S, S) 余弦相似度矩阵
upper_triangle = sim_matrix[i < j]          # 只取上三角（排除对角线）

stats:
  mean, std, min, max, median
  percentiles: p5, p25, p75, p95
```

**主指标**: `std`（标准差，越高越好 --> 区分度越强）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_size` | `5000` | 采样数量 |

**质量阈值**:

| 等级 | 条件 |
|------|------|
| Excellent | std >= 0.25 **且** 0.1 <= mean <= 0.3 |
| Good | std >= 0.20 **且** 0.1 <= mean <= 0.4 |
| Acceptable | 其他 |

**举例**: 5000 个 embedding
- mean=0.22, std=0.30 --> Excellent: 相似度分布广且中心合理
- mean=0.85, std=0.05 --> Poor: 所有向量高度相似，几乎没有区分力
- mean=0.15, std=0.22 --> Good: 区分力不错

---

### 5. Effective Dimension

**目标**: 衡量 embedding 空间有多少维度真正承载了信息（通过 PCA）。

**计算方法**:

```
# 采样并中心化
centered = sample - mean(sample)

# SVD 分解
_, S, _ = SVD(centered)
variance_explained = cumsum(S^2) / sum(S^2)

# 达到 threshold 方差所需的维度数
dim_90 = min(k: variance_explained[k] >= 0.90)
dim_95 = min(k: variance_explained[k] >= 0.95)
dim_99 = min(k: variance_explained[k] >= 0.99)

utilization_ratio = dim_95 / total_dim

# 参与率 (Participation Ratio)
eigenvalues = S^2 / sum(S^2)
participation_ratio = 1 / sum(eigenvalues^2)

# 谱衰减
top_10_ratio = sum(S[:10]^2) / sum(S^2)
```

**主指标**: `utilization_ratio = dim_95 / D`（值域 [0, 1]，越高越好）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_size` | `10000` | PCA 采样数 |
| `variance_thresholds` | `[0.90, 0.95, 0.99]` | 要报告的方差百分比 |

**质量阈值**:

| 等级 | 阈值 |
|------|------|
| Excellent | >= 0.70 |
| Good | >= 0.50 |
| Acceptable | >= 0.30 |
| Poor | < 0.30 |

**举例**: 768 维 embedding
- dim_95=600, utilization_ratio=0.78 --> Excellent: 78% 维度承载信息
- dim_95=150, utilization_ratio=0.20 --> Poor: 仅 20% 维度有用，严重冗余
- participation_ratio=350: 估计有 350 个"有效维度"

---

### 6. Semantic ID Collision

**目标**: 衡量不同内容被映射到相同 SID 前缀的碰撞程度，并报告桶大小分布（直接影响召回候选池大小）。

**计算方法**:

```
# 主指标: 完整 SID 碰撞率
collision_rate = 1 - (n_unique_sids / n_total)

# 逐前缀深度 (3 层为例)
depth=1: collision_rate_1 = 1 - unique("a_*_*") / n_total
depth=2: collision_rate_2 = 1 - unique("a_b_*") / n_total    ← 召回场景关注此层
depth=3: collision_rate_3 = 1 - unique("a_b_c") / n_total

# 每个 depth 额外报告桶大小分布:
prefix_stats[depth] = {
  n_unique_prefix, collision_rate,
  avg_items, min, max, p50, p90, p99,   ← 召回候选池大小分布
  le_1, le_2, le_5, le_10, ..., gt_500  ← 桶大小分箱计数
}
```

**layer_values**: 各 depth 的 collision rate

**主指标**: `collision_rate`（值域 [0, 1]，越低越好）

**质量阈值**:

| 等级 | 阈值 |
|------|------|
| Excellent | <= 1% |
| Good | <= 5% |
| Acceptable | <= 15% |
| Poor | > 15% |

**召回场景举例**: 285K 内容，3 层 64 clusters，用 `a_b_*` 做召回
- depth=2 prefix_stats:
  - n_unique=2803, avg=101.8, p50=68, p90=210, p99=485, max=920
  - 含义: 每个 d2 前缀平均覆盖 ~102 个内容，50% 的桶 <= 68 个，1% 的热门桶接近 500
  - 召回时: 查 1 个 d2 前缀就能拿到 ~100 个候选，top-K 从中选
- depth=1 prefix_stats:
  - n_unique=64, avg=4457, p50=4200, max=6800
  - 含义: L1 层只是粗分类，每个桶太大不适合直接做召回

---

### 7. Cluster Balance

**目标**: 衡量 SID 分布的均衡性（Gini 系数），按前缀深度分层统计。

**计算方法**:

```
# 主指标: 完整 SID 的 Gini
sid_counts = Counter(all_semantic_ids)
Gini_full = gini(sid_counts.values())

# 逐前缀深度 Gini (3 层为例)
depth=1: Gini over "a_*_*" prefix counts
depth=2: Gini over "a_b_*" prefix counts
depth=3: Gini over "a_b_c" full SID counts (= Gini_full)

Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
  0 = 完全均匀    1 = 极端不均匀
```

**主指标**: `gini`（值域 [0, 1]，越低越好）

**layer_values**: 各 depth 的 Gini

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | `50000` | 分批处理大小 |

**质量阈值**:

| 等级 | 阈值 |
|------|------|
| Excellent | <= 0.15 |
| Good | <= 0.25 |
| Acceptable | <= 0.40 |
| Poor | > 0.40 |

**举例**: 160K 内容，3 层 1024 clusters
- depth=1: gini=0.05 (1024 个 L1 前缀分布很均匀)
- depth=2: gini=0.18 (L1+L2 组合有些不均)
- depth=3: gini=0.10 (完整 SID 分布还行)
- 通过对比各 depth 的 Gini 可以看出不均衡发生在哪一层的展开

---

## 二、Behavior Metrics（行为指标，需要用户行为数据）

### 行为数据格式

```python
behavior_data = {
    'uid': np.array([...]),            # 用户 ID
    'iid': np.array([...]),            # 内容 ID (content_id)
    'action_bitmap': np.array([...]),  # 行为位图
    'first_ts': np.array([...]),       # 首次交互时间戳
}

# action_bitmap 定义:
#   bit 0  (1)            click
#   bit 1  (2)            like
#   bit 2  (4)            share
#   bit 3  (8)            follow
#   bit 31 (-2147483648)  negative_feedback
#
# 判断逻辑:
#   action_bitmap > 0  → 正向行为 (click/like/share/follow)
#   action_bitmap < 0  → 负反馈 (negative_feedback, bit 31 符号位)
```

---

### 8. User Semantic Consistency

**目标**: 同一用户喜欢的内容，其 SID 是否比随机更相似（验证 SID 是否编码了语义偏好）。

**计算方法**:

```
1. 对每个用户，收集所有 action > 0 的内容 (正向交互)
2. 筛选正向交互 >= min_positive_items 的用户
3. 对每个用户，计算其内容 SID 的平均 Jaccard 相似度:
   - SID "12_34_56" 解析为 token 序列 ("12", "34", "56")
   - Jaccard(a, b) = 逐层匹配数 / 层数
     例: Jaccard("12_34_56", "12_34_78") = 2/3 = 0.667
   - 用户内所有 pair 的平均相似度 → user_sim
4. 计算随机 baseline: 随机采样相同数量内容，计算平均相似度 → random_sim
5. Lift = (user_sim - random_sim) / random_sim
```

**主指标**: `lift_over_random`（越高越好，说明 SID 编码了用户偏好）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_positive_items` | `3` | 用户最少正向交互数 |
| `max_users` | `10000` | 最多采样用户数 |

**质量阈值**:

| 等级 | 阈值 |
|------|------|
| Excellent | >= 30% lift |
| Good | >= 20% lift |
| Acceptable | >= 10% lift |
| Poor | < 10% lift |

**举例**:
- 用户 A 喜欢了 SID: "12_34_56", "12_34_78", "12_50_90"
  - pair 相似度: (2/3, 1/3, 1/3) → user_sim = 0.444
- 随机 baseline: 三个随机 SID 平均相似度 0.15
- lift = (0.444 - 0.15) / 0.15 = 1.96 (196%) --> Excellent

---

### 9. Semantic Neighbor Hit Rate

**目标**: SID 前缀相同的内容（语义邻居），是否被相同用户群喜欢。

**计算方法**:

```
1. 构建 content → 喜欢它的用户集合 (positive_users)
2. 构建 SID前缀 → 内容列表 (prefix_layers 层前缀)
   例: prefix_layers=2, SID "12_34_56" → 前缀 "12_34"
3. 对每个内容 C:
   - 找 SID 前缀相同的邻居 (不含自身)
   - 对每个邻居 N:
     如果 positive_users(C) ∩ positive_users(N) 非空 → hit
   - hit_rate(C) = n_hits / n_neighbors
4. 最终指标 = mean(hit_rate) across all contents
```

**主指标**: `mean_hit_rate`（值域 [0, 1]，越高越好）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prefix_layers` | `2` | 前 N 层作为"相近"定义 |
| `max_items` | `5000` | 最多评估内容数 |

**质量阈值**:

| 等级 | 阈值 |
|------|------|
| Excellent | >= 15% |
| Good | >= 10% |
| Acceptable | >= 5% |
| Poor | < 5% |

**举例**:
- 内容 A 的 SID="12_34_56"，前缀 "12_34"
- 邻居: B("12_34_78"), C("12_34_90")
- 喜欢 A 的用户: {u1, u2, u3}
- 喜欢 B 的用户: {u2, u5} → 有交集 → hit
- 喜欢 C 的用户: {u8, u9} → 无交集 → miss
- hit_rate(A) = 1/2 = 0.50

---

### 10. Embedding-Behavior Correlation

**目标**: embedding 空间的余弦相似度 与 用户行为的重叠度 是否正相关。

**计算方法**:

```
1. 构建 content → positive_users 集合 (action > 0 且 |users| >= 5)
2. 随机采样 n_pairs 个内容对 (c1, c2)
3. 对每对:
   - emb_sim = cosine(embedding[c1], embedding[c2])
   - user_jaccard = |users(c1) ∩ users(c2)| / |users(c1) ∪ users(c2)|
4. Spearman 相关系数 = spearmanr(emb_similarities, user_overlaps)
```

**主指标**: `spearman_correlation`（值域 [-1, 1]，越高越好）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_pairs` | `10000` | 采样内容对数 |

**质量阈值**:

| 等级 | 阈值 |
|------|------|
| Excellent | >= 0.30 |
| Good | >= 0.20 |
| Acceptable | >= 0.10 |
| Poor | < 0.10 |

**举例**:
- 10000 对内容，embedding 相似度 vs 用户 Jaccard 重叠
- correlation = 0.35, p_value = 1e-100 --> Excellent: embedding 距离和用户偏好高度一致
- correlation = 0.05, p_value = 0.2 --> Poor: 几乎无关

---

### 11. Positive-Negative Separation

**目标**: 用户喜欢的内容(positive) 在 embedding 空间是否比不喜欢的(negative) 更聚集。

**计算方法**:

```
1. 对每个用户:
   - positive_items: action > 0 的内容
   - negative_items: action < 0 的内容 (negative feedback)
   - 筛选两者都 >= min_items 的用户
2. 计算:
   pos_embs = L2_normalize(embeddings[positive_items])
   neg_embs = L2_normalize(embeddings[negative_items])

   pos_dist = 1 - mean(pos_embs @ pos_embs.T)      # 正-正 平均距离
   neg_dist = 1 - mean(pos_embs @ neg_embs.T)       # 正-负 平均距离

   separation = (neg_dist - pos_dist) / neg_dist
3. 最终指标 = mean(separation) across users
```

**主指标**: `mean_separation`（越高越好，说明正负样本有效分离）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_items` | `2` | 用户最少正/负交互数 |
| `max_users` | `5000` | 最多采样用户数 |

**质量阈值**:

| 等级 | 阈值 |
|------|------|
| Excellent | >= 15% |
| Good | >= 10% |
| Acceptable | >= 5% |
| Poor | < 5% |

**举例**: 用户 U
- 喜欢内容 {A, B, C}，不喜欢 {X, Y}
- pos_dist = 0.3 (喜欢的内容之间距离 0.3)
- neg_dist = 0.5 (喜欢 vs 不喜欢之间距离 0.5)
- separation = (0.5 - 0.3) / 0.5 = 0.40 --> Excellent: 正负样本充分分离

---

### 12. Semantic ID Prediction (NTP)

**目标**: 用历史交互序列预测下一个 item 的 Semantic ID token，验证 SID 是否蕴含可被序列模型学习的用户行为模式。

**模型架构**:

```
AutoregressiveNTPModel:
  - Token Embedding: nn.Embedding(n_clusters, embed_dim=64)
  - Position Embedding: nn.Embedding(max_len, embed_dim=64)
  - Transformer Decoder: 2 层, 4 heads, FFN=256
  - Output Projection: Linear(64, n_clusters)
```

**生成过程** (Beam Search):

```
输入: 用户历史 k 个 item 的所有 tokens → 长度 = k * n_layers

Step 1: [历史 3k tokens]                        → 预测 next_L1
Step 2: [历史 3k tokens, next_L1]               → 预测 next_L2
Step 3: [历史 3k tokens, next_L1, next_L2]      → 预测 next_L3

Beam Search: 每步保留 top beam_size 个候选
```

**训练方式**:

```
- Teacher Forcing: 训练时目标 token 直接作为输入
- Loss: sum of cross_entropy(logits_i, target_i) for each layer
- 数据划分: 按时间全局排序，前 80% 训练，后 20% 评估
- 训练内用户顺序 shuffle，用户内保持时间序列
```

**评估指标**（全部按前缀深度统计）:

| 指标 | 说明 |
|------|------|
| Perplexity | exp(avg_loss / n_layers)，随机基线 = n_clusters |
| Depth Acc (beam) | 前缀深度准确率：depth=d 要求 L1..Ld **全部**正确 |
| Depth Hit@5 | Teacher Forcing 下前缀深度 top-5 命中：depth=d 要求 L1..Ld 全部 hit |
| Depth Hit@10 | 同上，top-10 |

```
# Beam Search prefix-depth accuracy
depth=1: L1 正确                        (最宽松)
depth=2: L1 且 L2 都正确
depth=3: L1+L2+L3 全正确                (最严格，= all_correct)

# Teacher Forcing prefix-depth Hit@K
depth=1: L1 在 top-K 中
depth=2: L1 在 top-K 中 且 L2 在 top-K 中
depth=3: L1+L2+L3 全在 top-K 中
```

**主指标**: `perplexity`（越低越好；随机基线 = n_clusters）

**layer_values**: 各 depth 的 beam search 准确率

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_items` | `10` | 历史序列长度 (item 数) |
| `epochs` | `10` | 训练轮数 |
| `batch_size` | `512` | 批大小 |
| `beam_size` | `5` | Beam Search 宽度 |
| `sample_users` | `50000` | 最多采样用户数 |
| `device` | `cuda` | 训练设备 |

**质量阈值**:

| 等级 | Perplexity 阈值 |
|------|------|
| Excellent | <= 50 |
| Good | <= 100 |
| Acceptable | <= 150 |
| Poor | > 150 |

**举例**: n_clusters=256, 3 层
- perplexity=40 --> Excellent: 远低于随机基线 256
- depth_acc_beam = [0.35, 0.12, 0.05]: depth=1 有 35% 前缀正确，depth=3 仅 5% 全对
- depth_hit@10 = [0.70, 0.35, 0.15]: 放宽到 top-10 后，前缀命中率显著提升

---

## 指标总览

| 指标 | 主值 | 方向 | 需要模型 | 需要 SID | 需要行为 |
|------|------|------|---------|---------|---------|
| Reconstruction Loss | normalized_loss | 越低越好 | Yes | No | No |
| Codebook Utilization | space_utilization | -- | No | Yes | No |
| Token Entropy | normalized_entropy | 越高越好 | No | Yes | No |
| Cosine Similarity | std | 越高越好 | No | No | No |
| Effective Dimension | utilization_ratio | 越高越好 | No | No | No |
| Semantic ID Collision | collision_rate | 越低越好 | No | Yes | No |
| Cluster Balance | gini | 越低越好 | No | Yes | No |
| User Semantic Consistency | lift_over_random | 越高越好 | No | Yes | Yes |
| Semantic Neighbor Hit Rate | mean_hit_rate | 越高越好 | No | Yes | Yes |
| Embedding-Behavior Correlation | spearman_corr | 越高越好 | No | No | Yes |
| Positive-Negative Separation | mean_separation | 越高越好 | No | No | Yes |
| Semantic ID Prediction | perplexity | 越低越好 | No | Yes | Yes |
