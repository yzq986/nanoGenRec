# Semantic ID Next Token Prediction (NTP) Metric

## Overview

基于 OneRec 论文 (arxiv 2506.13695) 实现的 Semantic ID 预测评估指标。

**核心思想**: 用用户历史交互的 k 个 item 的 Semantic ID tokens，预测下一个 item 的 Semantic ID。

---

## 数据设计

### 1. 原始数据

来自 `export_user_behavior.py` 导出的用户行为数据：

```
┌──────────┬──────────┬───────────────┬────────────┐
│ uid      │ iid      │ action_bitmap │ first_ts   │
├──────────┼──────────┼───────────────┼────────────┤
│ user_A   │ item_1   │ 3             │ 1711800000 │
│ user_A   │ item_2   │ 1             │ 1711800100 │
│ user_B   │ item_5   │ 7             │ 1711800050 │
│ ...      │ ...      │ ...           │ ...        │
└──────────┴──────────┴───────────────┴────────────┘
```

- `action_bitmap > 0`: 正向行为 (click, like, share, etc.)
- `first_ts`: 首次交互时间戳

### 2. Semantic ID 格式

RKMeans 生成的 3 层 Semantic ID：

```
"12_34_56" → [12, 34, 56]  # Layer 1, Layer 2, Layer 3
```

每层 256 个 clusters (可配置)。

### 3. 样本构建

**滑动窗口** (n_items=10)：

```
用户序列: [item_1, item_2, ..., item_15] (按时间排序)

样本 1: input = [item_1 ~ item_10], target = item_11
样本 2: input = [item_2 ~ item_11], target = item_12
样本 3: input = [item_3 ~ item_12], target = item_13
...
```

**Token 格式**：

```
input_tokens:  [L1,L2,L3, L1,L2,L3, ..., L1,L2,L3]  # 10 items × 3 layers = 30 tokens
                item_1    item_2        item_10

target_tokens: [L1, L2, L3]  # 下一个 item 的 3 个 tokens
target_ts:     1711800500    # 用于全局时间排序
```

### 4. 时间排序 & Split

**关键**: 避免时间穿透 (data leakage)

```
所有样本按 target_ts 全局排序:

时间线:
|<---------- Train (80%) ---------->|<--- Eval (20%) --->|
样本_1  样本_2  ...  样本_M          样本_M+1  ...  样本_N
T_start                             T_split              T_end
```

- **Train**: 时间较早的 80% 样本，shuffle 后训练
- **Eval**: 时间较晚的 20% 样本，只评估不训练

---

## 模型架构

### AutoregressiveNTPModel

```
Input: 30 tokens (10 items × 3 layers)
       ↓
Token Embedding + Position Embedding
       ↓
Transformer Decoder (Causal Self-Attention) × 2 layers
       ↓
Output Projection → 256 classes

Generation (自回归):
  Step 1: [30 tokens] → predict L1
  Step 2: [30 tokens, L1] → predict L2
  Step 3: [30 tokens, L1, L2] → predict L3
```

**参数**:
- `embed_dim`: 64
- `n_heads`: 4
- `n_transformer_layers`: 2
- Total: ~170K params

### Beam Search

推理时使用 Beam Search (beam_size=5):

```
P(item) = P(L1) × P(L2|L1) × P(L3|L1,L2)
```

---

## 评估指标

| 指标 | 定义 | 意义 |
|------|------|------|
| **Perplexity** (主指标) | exp(avg_loss / n_layers) | 越低越好，random baseline = 256 |
| **All Correct** | 3 层全对的比例 | Beam Search 输出 |
| **Layer Acc** | 每层单独的准确率 | Beam Search 输出 |
| **Hit@K** | 正确答案在 top-K 中的比例 | Teacher Forcing 下计算 |

### 质量判断

```python
thresholds = {
    'excellent': 50,   # PPL <= 50
    'good': 100,       # PPL <= 100
    'acceptable': 150, # PPL <= 150
    'poor': > 150
}
```

Random baseline PPL = 256 (n_clusters)

---

## 使用方法

### 命令行

```bash
# 只运行 SID Prediction
python demo_eval_all.py --only-sid --models qwen3-0.6b

# 调整参数
python demo_eval_all.py --only-sid \
    --models qwen3-0.6b \
    --sid_sample_users 50000  # 采样用户数
```

### 代码调用

```python
from metrics import SemanticIDPredictionMetric

metric = SemanticIDPredictionMetric()
result = metric.compute(
    embeddings=embeddings,
    semantic_ids=semantic_ids,
    behavior_data=behavior_data,
    content_id_to_idx=content_id_to_idx,
    n_items=10,        # 历史 item 数
    beam_size=5,       # Beam Search 宽度
    sample_users=50000,
    device='cuda',
)

print(f"Perplexity: {result.value}")
print(f"Details: {result.details}")
```

---

## 输出示例

```
Computing semantic_id_prediction...
  Users: 20000, Total samples: 516330
  Train: 413064, Eval: 103266 (by time split)
  n_layers: 3, n_clusters: 256
  Model: 168,640 params
  Training...
    Train progress: 10.0% | Loss: 2.8123
    Train progress: 20.0% | Loss: 2.5012
    ...
  Evaluating on 103266 samples (beam=5)...
  Results:
    Perplexity: 3.45 (random: 256)
    All Correct (beam): 0.1823
    Layer Acc (beam): ['0.3912', '0.8834', '0.6521']
    Hit@10: ['0.9812', '0.9956', '0.9971']
```

---

## 设计考量

### 为什么按时间 Split？

| 方案 | 问题 |
|------|------|
| 随机 split 用户 | 不同用户的序列可能时间交错，有泄露 |
| 用户内 split | 无跨用户学习，但滚动训练时还是可能穿透 |
| **全局时间 split** | 严格保证 eval 在 train 之后，无泄露 |

### 为什么用 Perplexity？

- OneRec 论文使用 Cross-Entropy Loss / Perplexity
- 推荐场景不适合用 Accuracy（太严格）
- Perplexity 可解释：模型的"困惑程度"，random = n_clusters

### 为什么 1 Epoch？

- 严格时间 split 下，多 epoch 对 train 数据过拟合
- 单 epoch 更接近 online learning 场景
- 实验显示 1 epoch 足够收敛

---

## 参考

- [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2506.13695)
- RKMeans (Residual K-Means) 实现: `rkmeans_stage2_train_v2.py`
