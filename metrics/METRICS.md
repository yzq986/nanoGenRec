# Embedding & Semantic ID evaluation index system

[English](METRICS.md) | [Chinese](METRICS.zh.md)

> Reference: OneRec (arxiv 2506.13695)

This module provides two categories of **12 indicators** in total, which are used to evaluate the quality of the embedding model and the quality of Semantic ID after RKMeans quantification.

---

## Table of contents

- [1. Intrinsic Metrics (intrinsic indicators, no behavioral data required)] (#一intrinsic-metrics intrinsic indicators do not require behavioral data)
  - [1. Reconstruction Loss](#1-reconstruction-loss)
  - [2. Codebook Utilization](#2-codebook-utilization)
  - [3. Token Entropy](#3-token-entropy)
  - [4. Cosine Similarity Distribution](#4-cosine-similarity-distribution)
  - [5. Effective Dimension](#5-effective-dimension)
  - [6. Semantic ID Collision](#6-semantic-id-collision)
  - [7. Cluster Balance](#7-cluster-balance)
- [2. Behavior Metrics (behavior indicators, user behavior data is required)] (#二behavior-metrics behavior indicators require user behavior data)
  - [8. User Semantic Consistency](#8-user-semantic-consistency)
  - [9. Semantic Neighbor Hit Rate](#9-semantic-neighbor-hit-rate)
  - [10. Embedding-Behavior Correlation](#10-embedding-behavior-correlation)
  - [11. Positive-Negative Separation](#11-positive-negative-separation)
  - [12. Semantic ID Prediction (NTP)](#12-semantic-id-prediction-ntp)

---

## 1. Intrinsic Metrics (intrinsic indicators, no behavioral data required)

### 1. Reconstruction Loss

**Goal**: Measure the accuracy of RKMeans quantization, that is, how much the original embedding can be restored after being quantized.

**Calculation method** (Same as `generate_semantic_ids` of `rkmeans_stage2_train_v2.py`):

```
For L-layer RKMeans:
# Step 0: Only perform L2 normalization once on the original input (layer 0 only)
  if normalize_residuals:
    residual = L2_normalize(embedding)       # ||residual|| = 1
  else:
    residual = embedding

input_for_loss = residual # Save the normalized input for calculating loss

# Step 1..L: Quantize layer by layer, the residual will not be re-normalized
  for layer_i in [1..L]:
    assignment = argmin ||residual - centroid||^2
    reconstruction_i = centroid[assignment] # Use centroid directly without scaling
    residual = residual - reconstruction_i # The residual retains the original scale

  x_hat = sum(reconstruction_1, ..., reconstruction_L)
  total_loss = mean(||input_for_loss - x_hat||^2)
  normalized_loss = total_loss / mean(||input_for_loss||^2)
```

> **Note**: `normalize_residuals=True` only performs L2 normalization on the input once in layer 0, and the residuals of subsequent layers will not be re-normalized.
> This is completely consistent with the logic of training and inference (`generate_semantic_ids`).

**Main indicator**: `normalized_loss` (value range [0, +inf), the smaller the better)

| Parameter | 默认Value | Description |
|------|--------|------|
| `normalize_residuals` | `True` | 仅对 layer 0 Input做 L2 归一化（后续层不再归一化） |
| `chunk_size` | `50000` | 分批处理大小 |

**Quality Threshold**:

| 等级 | 阈Value |
|------|------|
| Excellent | <= 0.05 |
| Good | <= 0.10 |
| Acceptable | <= 0.20 |
| Poor | > 0.20 |

**Example**: Assume embedding dimension D=768, 100,000 samples, 3 layers of RKMeans (256 centroids per layer)
- normalized_loss = 0.03 --> Excellent: only 3% information loss
- normalized_loss = 0.15 --> Acceptable: 15% loss, it is recommended to increase the number of layers or clusters

---

### 2. Codebook Utilization

**Goal**: Measure the utilization of SID space at each prefix depth.

**Calculation method**:

```
# Prefix-by-prefix depth statistics (layer 3 as an example)
depth=1: unique("a_*_*") / N^1 # L1 prefix utilization
depth=2: unique("a_b_*") / N^2 # L1+L2 prefix utilization
depth=3: unique("a_b_c") / N^3 # Complete SID utilization

layer_values = [util_depth1, util_depth2, util_depth3]
```

**Main indicator**: `space_utilization = n_unique_full_sids / N^L` (value range [0, 1])

**layer_values**: `n_unique_prefix / N^depth` for each depth

| Parameter | 默认Value | Description |
|------|--------|------|
| `chunk_size` | `50000` | 分批处理大小 |

**Quality Threshold**: No fixed threshold (depends on n_items / N^L ratio), status is `unknown`

**Example**: 160K content, 3 layers 1024 clusters
- depth=1: 1020 unique / 1024 = 0.9961 (99.6% L1 cluster is used)
- depth=2: 150,000 unique / 1,048,576 = 0.1431 (14.3% L1+L2 combination is used)
- depth=3: 158,000 unique / 1,073,741,824 = 0.0001 (full SID space is huge)
- The shallower the layer, the higher the utilization rate. The deeper the layer, the lower it is due to the combined explosion, which is a normal phenomenon.

---

### 3. Token Entropy

**Goal**: Measure the uniformity of SID distribution and stratify statistics by prefix depth.

**Calculation method**:

```
# Primary indicator: normalized entropy of complete SID
sid_counts = Counter(all_semantic_ids)
H_full = -sum( p(sid) * log2(p(sid)) )
normalized = H_full / log2(N_total)

# Prefix-by-prefix depth entropy (3 layers as an example)
depth=1: prefixes = ["a_*_*"]
  H_1 = -sum( p(prefix) * log2(p(prefix)) )
  normalized_1 = H_1 / log2(N) # Maximum entropy = log2(N^1)

depth=2: prefixes = ["a_b_*"]
  H_2 = -sum( p(prefix) * log2(p(prefix)) )
  normalized_2 = H_2 / log2(N^2) # Maximum entropy = log2(N^2)

depth=3: prefixes = ["a_b_c"] (= full SID)
  H_3 = H_full
  normalized_3 = H_3 / log2(N^3) # Maximum entropy = log2(N^3)
```

**Main indicator**: `normalized_entropy = H_full / log2(N_total)` (value range [0, 1], the higher the better)

**layer_values**: `H_depth / log2(N^depth)` for each depth

| Parameter | 默认Value | Description |
|------|--------|------|
| `chunk_size` | `50000` | 分批处理大小 |

**Quality Threshold**:

| 等级 | 阈Value |
|------|------|
| Excellent | >= 0.95 |
| Good | >= 0.90 |
| Acceptable | >= 0.80 |
| Poor | < 0.80 |

**Example**: 160K content, 3 layers 1024 clusters
- depth=1: H=9.95 / log2(1024)=10.0 → normalized=0.995 (L1 prefix is very uniform)
- depth=2: H=16.8 / log2(1024^2)=20.0 → normalized=0.840 (L1+L2 combination is somewhat concentrated)
- depth=3: H=17.1 / log2(1024^3)=30.0 → normalized=0.570 (the complete SID space is huge and the concentration is higher)
- Main indicator normalized = 17.1 / log2(160000) = 0.989

---

### 4. Cosine Similarity Distribution

**Goal**: Check the discriminative ability of embedding - a good embedding should have moderate mean and large std (discriminative).

**Calculation method**:

```
# Sample sample_size embeddings
sample_norm = L2_normalize(sample)
sim_matrix = sample_norm @ sample_norm.T # (S, S) cosine similarity matrix
upper_triangle = sim_matrix[i < j] # Take only the upper triangle (exclude diagonals)

stats:
  mean, std, min, max, median
  percentiles: p5, p25, p75, p95
```

**Main indicator**: `std` (standard deviation, the higher the better --> the stronger the discrimination)

| Parameter | 默认Value | Description |
|------|--------|------|
| `sample_size` | `5000` | 采样数量 |

**Quality Threshold**:

| 等级 | 条件 |
|------|------|
| Excellent | std >= 0.25 **且** 0.1 <= mean <= 0.3 |
| Good | std >= 0.20 **且** 0.1 <= mean <= 0.4 |
| Acceptable | 其他 |

**Example**: 5000 embeddings
- mean=0.22, std=0.30 --> Excellent: The similarity is widely distributed and the center is reasonable
- mean=0.85, std=0.05 --> Poor: All vectors are highly similar and have almost no discriminating power
- mean=0.15, std=0.22 --> Good: good discrimination

---

### 5. Effective Dimension

**Goal**: Measure how many dimensions of the embedding space actually carry information (via PCA).

**Calculation method**:

```
# Sampling and centralizing
centered = sample - mean(sample)

# SVD decomposition
_, S, _ = SVD(centered)
variance_explained = cumsum(S^2) / sum(S^2)

# Number of dimensions required to reach threshold variance
dim_90 = min(k: variance_explained[k] >= 0.90)
dim_95 = min(k: variance_explained[k] >= 0.95)
dim_99 = min(k: variance_explained[k] >= 0.99)

utilization_ratio = dim_95 / total_dim

# Participation Ratio
eigenvalues = S^2 / sum(S^2)
participation_ratio = 1 / sum(eigenvalues^2)

# Spectral attenuation
top_10_ratio = sum(S[:10]^2) / sum(S^2)
```

**Main indicator**: `utilization_ratio = dim_95 / D` (value range [0, 1], the higher the better)

| Parameter | 默认Value | Description |
|------|--------|------|
| `sample_size` | `10000` | PCA 采样数 |
| `variance_thresholds` | `[0.90, 0.95, 0.99]` | 要报告的方差百分比 |

**Quality Threshold**:

| 等级 | 阈Value |
|------|------|
| Excellent | >= 0.70 |
| Good | >= 0.50 |
| Acceptable | >= 0.30 |
| Poor | < 0.30 |

**Example**: 768-dimensional embedding
- dim_95=600, utilization_ratio=0.78 --> Excellent: 78% dimensions carry information
- dim_95=150, utilization_ratio=0.20 --> Poor: only 20% of dimensions are useful, serious redundancy
- participation_ratio=350: estimated 350 "valid dimensions"

---

### 6. Semantic ID Collision

**Goal**: Measure the degree of collision of different content being mapped to the same SID prefix, and report the bucket size distribution (directly affecting the recall candidate pool size).

**Calculation method**:

```
# Primary indicator: Complete SID collision rate
collision_rate = 1 - (n_unique_sids / n_total)

# Depth by prefix (layer 3 as an example)
depth=1: collision_rate_1 = 1 - unique("a_*_*") / n_total
depth=2: collision_rate_2 = 1 - unique("a_b_*") / n_total ← Recall the scene and focus on this layer
depth=3: collision_rate_3 = 1 - unique("a_b_c") / n_total

# Additional reporting of bucket size distribution for each depth:
prefix_stats[depth] = {
  n_unique_prefix, collision_rate,
  avg_items, min, max, p50, p90, p99, ← recall candidate pool size distribution
  le_1, le_2, le_5, le_10, ..., gt_500 ← Bucket size bin count
}
```

**layer_values**: collision rate of each depth

**Main indicator**: `collision_rate` (value range [0, 1], the lower the better)

**Quality Threshold**:

| 等级 | 阈Value |
|------|------|
| Excellent | <= 1% |
| Good | <= 5% |
| Acceptable | <= 15% |
| Poor | > 15% |

**Example of recall scenario**: 285K content, 3 layers and 64 clusters, using `a_b_*` for recall
- depth=2 prefix_stats:
  - n_unique=2803, avg=101.8, p50=68, p90=210, p99=485, max=920
  - Meaning: Each d2 prefix covers ~102 contents on average, 50% of buckets <= 68, 1% of popular buckets are close to 500
  - When recalling: Just look up 1 d2 prefix to get ~100 candidates, and choose top-K from them
- depth=1 prefix_stats:
  - n_unique=64, avg=4457, p50=4200, max=6800
  - Meaning: The L1 layer is only a rough classification, and each bucket is too large to be directly recalled

---

### 7. Cluster Balance

**Goal**: Measure the balance of SID distribution (Gini coefficient), stratified statistics by prefix depth.

**Calculation method**:

```
# Primary indicator: Gini of complete SID
sid_counts = Counter(all_semantic_ids)
Gini_full = gini(sid_counts.values())

# Prefix-by-prefix depth Gini (3 layers as an example)
depth=1: Gini over "a_*_*" prefix counts
depth=2: Gini over "a_b_*" prefix counts
depth=3: Gini over "a_b_c" full SID counts (= Gini_full)

Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
  0 = perfectly uniform 1 = extremely uneven
```

**Main indicator**: `gini` (value range [0, 1], the lower the better)

**layer_values**: Gini for each depth

| Parameter | 默认Value | Description |
|------|--------|------|
| `chunk_size` | `50000` | 分批处理大小 |

**Quality Threshold**:

| 等级 | 阈Value |
|------|------|
| Excellent | <= 0.15 |
| Good | <= 0.25 |
| Acceptable | <= 0.40 |
| Poor | > 0.40 |

**Example**: 160K content, 3 layers 1024 clusters
- depth=1: gini=0.05 (1024 L1 prefixes are evenly distributed)
- depth=2: gini=0.18 (L1+L2 combination is somewhat uneven)
- depth=3: gini=0.10 (full SID distribution is OK)
- By comparing the Gini of each depth, you can see at which level of expansion the imbalance occurs.

---

## 2. Behavior Metrics (behavior indicators, requiring user behavior data)

### Behavioral data format

```python
behavior_data = {
'uid': np.array([...]), # User ID
'iid': np.array([...]), # Content ID (content_id)
'action_bitmap': np.array([...]), #Action bitmap
'first_ts': np.array([...]), # First interaction timestamp
}

# action_bitmap definition:
# bit 0 (1) click
# bit 1 (2) like
# bit 2 (4) share
# bit 3 (8) follow
# bit 31 (-2147483648) negative_feedback
#
# Judgment logic:
# action_bitmap > 0 → positive action (click/like/share/follow)
# action_bitmap < 0 → negative feedback (negative_feedback, bit 31 sign bit)
```

---

### 8. User Semantic Consistency

**Goal**: Whether the SIDs of content liked by the same user are more similar than random (verify whether the SID encodes semantic preferences).

**Calculation method**:

```
1. For each user, collect all content with action > 0 (forward interaction)
2. Filter users with positive interactions >= min_positive_items
3. For each user, calculate the average Jaccard similarity of its content SID:
- SID "12_34_56" is parsed into token sequence ("12", "34", "56")
- Jaccard(a, b) = number of layer-by-layer matching / number of layers
Example: Jaccard("12_34_56", "12_34_78") = 2/3 = 0.667
- Average similarity of all pairs within the user → user_sim
4. Calculate random baseline: randomly sample the same amount of content and calculate the average similarity → random_sim
5. Lift = (user_sim - random_sim) / random_sim
```

**Main indicator**: `lift_over_random` (the higher the better, indicating that the SID encodes user preferences)

| Parameter | 默认Value | Description |
|------|--------|------|
| `min_positive_items` | `3` | 用户最少正向交互数 |
| `max_users` | `10000` | 最多采样用户数 |

**Quality Threshold**:

| 等级 | 阈Value |
|------|------|
| Excellent | >= 30% lift |
| Good | >= 20% lift |
| Acceptable | >= 10% lift |
| Poor | < 10% lift |

**Example**:
- User A liked SID: "12_34_56", "12_34_78", "12_50_90"
  - pair similarity: (2/3, 1/3, 1/3) → user_sim = 0.444
- Random baseline: average similarity of three random SIDs is 0.15
- lift = (0.444 - 0.15) / 0.15 = 1.96 (196%) --> Excellent

---

### 9. Semantic Neighbor Hit Rate

**Target**: Whether content with the same SID prefix (semantic neighbor) is liked by the same user group.

**Calculation method**:

```
1. Build content → the collection of users who like it (positive_users)
2. Build SID prefix → content list (prefix_layers layer prefix)
Example: prefix_layers=2, SID "12_34_56" → prefix "12_34"
3. For each content C:
- Find neighbors with the same SID prefix (excluding itself)
- For each neighbor N:
If positive_users(C) ∩ positive_users(N) is not empty → hit
   - hit_rate(C) = n_hits / n_neighbors
4. Final indicator = mean(hit_rate) across all contents
```

**Main indicator**: `mean_hit_rate` (value range [0, 1], the higher the better)

| Parameter | 默认Value | Description |
|------|--------|------|
| `prefix_layers` | `2` | 前 N 层作为"相近"定义 |
| `max_items` | `5000` | 最多Evaluation内容数 |

**Quality Threshold**:

| 等级 | 阈Value |
|------|------|
| Excellent | >= 15% |
| Good | >= 10% |
| Acceptable | >= 5% |
| Poor | < 5% |

**Example**:
- Content A has SID="12_34_56", prefix "12_34"
- Neighbors: B("12_34_78"), C("12_34_90")
- Users who like A: {u1, u2, u3}
- Users who like B: {u2, u5} → have intersection → hit
- Users who like C: {u8, u9} → no intersection → miss
- hit_rate(A) = 1/2 = 0.50

---

### 10. Embedding-Behavior Correlation

**Goal**: Whether the cosine similarity of the embedding space is positively correlated with the overlap of user behavior.

**Calculation method**:

```
1. Build content → positive_users collection (action > 0 and |users| >= 5)
2. Randomly sample n_pairs content pairs (c1, c2)
3. For each pair:
   - emb_sim = cosine(embedding[c1], embedding[c2])
   - user_jaccard = |users(c1) ∩ users(c2)| / |users(c1) ∪ users(c2)|
4. Spearman correlation coefficient = spearmanr(emb_similarities, user_overlaps)
```

**Main indicator**: `spearman_correlation` (value range [-1, 1], higher is better)

| Parameter | 默认Value | Description |
|------|--------|------|
| `n_pairs` | `10000` | 采样内容对数 |

**Quality Threshold**:

| 等级 | 阈Value |
|------|------|
| Excellent | >= 0.30 |
| Good | >= 0.20 |
| Acceptable | >= 0.10 |
| Poor | < 0.10 |

**Example**:
- 10000 pairs of content, embedding similarity vs user Jaccard overlap
- correlation = 0.35, p_value = 1e-100 --> Excellent: embedding distance is highly consistent with user preference
- correlation = 0.05, p_value = 0.2 --> Poor: almost irrelevant

---

### 11. Positive-Negative Separation

**Goal**: Whether the content that the user likes (positive) is more concentrated in the embedding space than the content that the user does not like (negative).

**Calculation method**:

```
1. For each user:
- positive_items: action > 0 content
- negative_items: content with action < 0 (negative feedback)
- Filter users who have both >= min_items
2. Calculation:
   pos_embs = L2_normalize(embeddings[positive_items])
   neg_embs = L2_normalize(embeddings[negative_items])

pos_dist = 1 - mean(pos_embs @ pos_embs.T) # Positive-positive average distance
   neg_dist = 1 - mean(pos_embs @ neg_embs.T) # Positive-negative average distance

separation = (neg_dist - pos_dist) / neg_dist
3. Final indicator = mean(separation) across users
```

**Main indicator**: `mean_separation` (the higher, the better, indicating that positive and negative samples are effectively separated)

| Parameter | 默认Value | Description |
|------|--------|------|
| `min_items` | `2` | 用户最少正/负交互数 |
| `max_users` | `5000` | 最多采样用户数 |

**Quality Threshold**:

| 等级 | 阈Value |
|------|------|
| Excellent | >= 15% |
| Good | >= 10% |
| Acceptable | >= 5% |
| Poor | < 5% |

**Example**: User U
- Like content {A, B, C}, dislike {X, Y}
- pos_dist = 0.3 (distance between likes 0.3)
- neg_dist = 0.5 (distance between likes vs dislikes 0.5)
- separation = (0.5 - 0.3) / 0.5 = 0.40 --> Excellent: Positive and negative samples are fully separated

---

### 12. Semantic ID Prediction (NTP)

**Goal**: Use historical interaction sequences to predict the Semantic ID token of the next item, and verify whether the SID contains user behavior patterns that can be learned by the sequence model.

**Model Architecture**:

```
AutoregressiveNTPModel:
  - Token Embedding: nn.Embedding(n_clusters, embed_dim=64)
  - Position Embedding: nn.Embedding(max_len, embed_dim=64)
- Transformer Decoder: 2 layers, 4 heads, FFN=256
  - Output Projection: Linear(64, n_clusters)
```

**Generation process** (Beam Search):

```
Input: all tokens of k items in user history → length = k * n_layers

Step 1: [History 3k tokens] → Predict next_L1
Step 2: [History 3k tokens, next_L1] → Predict next_L2
Step 3: [History 3k tokens, next_L1, next_L2] → Predict next_L3

Beam Search: retain top beam_size candidates at each step
```

**Training method**:

```
- Teacher Forcing: The target token is directly used as input during training
- Loss: sum of cross_entropy(logits_i, target_i) for each layer
- Data partitioning: Global sorting by time, first 80% training, last 20% evaluation
- Shuffle user order within training, maintain time series within users
```

**Evaluation indicators** (all statistics based on prefix depth):

| Metric | Description |
|------|------|
| Perplexity | exp(avg_loss / n_layers)，随机Baseline = n_clusters |
| Depth Acc (beam) | 前缀深度准确率：depth=d 要求 L1..Ld **全部**正确 |
| Depth Hit@5 | Teacher Forcing 下前缀深度 top-5 命Medium：depth=d 要求 L1..Ld 全部 hit |
| Depth Hit@10 | 同上，top-10 |

```
# Beam Search prefix-depth accuracy
depth=1: L1 correct (most relaxed)
depth=2: L1 and L2 are both correct
depth=3: L1+L2+L3 all correct (most strict, = all_correct)

# Teacher Forcing prefix-depth Hit@K
depth=1: L1 in top-K
depth=2: L1 is in top-K and L2 is in top-K
depth=3: L1+L2+L3 are all in top-K
```

**Main metric**: `perplexity` (lower is better; random baseline = n_clusters)

**layer_values**: beam search accuracy of each depth

| Parameter | 默认Value | Description |
|------|--------|------|
| `n_items` | `10` | 历史序列长度 (item 数) |
| `epochs` | `10` | Training轮数 |
| `batch_size` | `512` | 批大小 |
| `beam_size` | `5` | Beam Search 宽度 |
| `sample_users` | `50000` | 最多采样用户数 |
| `device` | `cuda` | Training设备 |

**Quality Threshold**:

| 等级 | Perplexity 阈Value |
|------|------|
| Excellent | <= 50 |
| Good | <= 100 |
| Acceptable | <= 150 |
| Poor | > 150 |

**Example**: n_clusters=256, 3 layers
- perplexity=40 --> Excellent: well below random baseline 256
- depth_acc_beam = [0.35, 0.12, 0.05]: depth=1 has 35% correct prefixes, depth=3 only 5% correct
- depth_hit@10 = [0.70, 0.35, 0.15]: After relaxing to top-10, the prefix hit rate is significantly improved

---

## Overview of indicators

| Metric | 主Value | Direction | 需要Model | 需要 SID | 需要行为 |
|------|------|------|---------|---------|---------|
| Reconstruction Loss | normalized_loss | 越Low越Good | Yes | No | No |
| Codebook Utilization | space_utilization | -- | No | Yes | No |
| Token Entropy | normalized_entropy | 越High越Good | No | Yes | No |
| Cosine Similarity | std | 越High越Good | No | No | No |
| Effective Dimension | utilization_ratio | 越High越Good | No | No | No |
| Semantic ID Collision | collision_rate | 越Low越Good | No | Yes | No |
| Cluster Balance | gini | 越Low越Good | No | Yes | No |
| User Semantic Consistency | lift_over_random | 越High越Good | No | Yes | Yes |
| Semantic Neighbor Hit Rate | mean_hit_rate | 越High越Good | No | Yes | Yes |
| Embedding-Behavior Correlation | spearman_corr | 越High越Good | No | No | Yes |
| Positive-Negative Separation | mean_separation | 越High越Good | No | No | Yes |
| Semantic ID Prediction | perplexity | 越Low越Good | No | Yes | Yes |
