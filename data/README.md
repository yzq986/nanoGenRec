# data/ — 数据加载与预处理

负责从 S3/Hive 加载原始数据，以及分布式 Embedding 编码。

## 文件说明

| 文件 | 说明 |
|------|------|
| `loaders.py` | 统一的 S3 / 本地数据加载函数集合 |
| `encode_distributed.py` | `torchrun` 多卡分布式文本编码 |
| `export_content.py` | PySpark 导出曝光内容 (文本 + 图片) 到 S3 |
| `export_behavior.py` | PySpark 天级别增量导出用户行为 bitmap 到 S3 |

## loaders.py

S3 和本地的数据加载/导出函数：

- `load_text_from_s3()` — 加载内容文本 (parquet)
- `load_old_embeddings_from_s3()` — 加载旧 Sentence-BERT embedding
- `load_exposed_iids()` — 加载曝光 item ID 列表
- `export_results_to_s3()` — 导出结果 (content_id, semantic_id, embedding) 到 S3
- `load_results_from_s3()` / `load_model_from_s3()` — 评测用数据加载
- `load_local_results()` / `load_local_model()` — 本地文件加载

## encode_distributed.py

基于 `torchrun` 的多 GPU 分布式编码：

```bash
torchrun --nproc_per_node=8 -m gr_demo.data.encode_distributed --model qwen3-0.6b
```

- 增量缓存: 自动跳过已编码的 content ID
- OOM 重试: CUDA OOM 时自动减半 batch size
- Rank 0 负责模型下载协调、缓存广播、结果合并

## export_content.py

PySpark 脚本，按日期范围一次性导出曝光内容到 S3：

- 输入: `DATE_KEY_START` ~ `DATE_KEY_END`
- 输出: `{S3_CONTENT_TEXT_EXPOSED}/{DATE_KEY_END}/` (文本 + 图片 URL)

## export_behavior.py

PySpark 脚本，天级别增量导出用户行为 bitmap 到 S3：

- 输入: `DATE_KEY_START` ~ `DATE_KEY_END`，逐天循环导出
- 输出: `{S3_USER_BEHAVIOR}/{date}/` 每天一个独立目录
- 20+ 种交互类型编码为 `action_bitmap`
- 支持单日模式 (cron/airflow 调度) 和范围模式 (补跑)

## 数据分布特征 (2026-01-25 ~ 2026-03-31)

基于 `experiments/scripts/analyze_data_distribution.py` 的分析结果。

### 用户行为分布 — 极度长尾

| 时间窗口 | 用户数 | 正向交互数 | Mean/User | P50 | P95 | P99 | Max |
|----------|-------|-----------|-----------|-----|-----|-----|-----|
| 7d (03-25~03-31) | 1.54M | 23.9M | 15.6 | 3 | 68 | 220 | 5,376 |
| 14d (03-18~03-31) | 2.51M | 53.1M | 21.2 | 3 | 92 | 331 | 9,063 |
| 31d (03-01~03-31) | 4.55M | 129.7M | 28.5 | 3 | 118 | 499 | 32,246 |
| 62d (02-01~03-31) | 7.29M | 261.8M | 35.9 | 3 | 138 | 669 | 46,223 |
| 66d (01-25~03-31) | 7.85M | 299.0M | 38.1 | 3 | 146 | 715 | 46,990 |

**关键特征**:
- **P50 恒定为 3**: 50% 用户只有 ≤3 次正向交互，分布极度右偏
- **少数重度用户贡献大量数据**: ~4% 用户贡献 ~50% 的交互量
- 用户数随时间窗口近似线性增长 (每月 ~2.5M 新增活跃用户)
- 每用户均值增长缓慢 (7d→66d 仅 15.6→38.1)，主要靠用户数驱动总量

### 序列截断影响 (max_seq_len=512 → 170 items/user)

NTP 训练每用户最多保留最近 170 个 item (max_seq_len=512, n_layers=3)。截断分析:

| 时间窗口 | 截断用户% | Items 丢失% | Raw Items | 有效 Items | 有效 Tokens |
|----------|----------|------------|-----------|-----------|------------|
| 7d | 1.5% | 14.5% | 23.9M | ~20.4M | ~61M |
| 14d | 2.6% | 25.4% | 53.1M | ~39.6M | ~119M |
| 31d | 3.6% | 38.9% | 129.7M | ~79.3M | ~238M |
| 62d | 4.2% | 48.5% | 261.8M | ~134.8M | ~404M |
| 66d | 4.4% | 50.4% | 299.0M | ~148.3M | ~445M |

> **有效 Tokens = 有效 Items × 3** (每个 item 由 3 个 SID token 表示)

**两个维度的不矛盾现象**:
- **用户维度**: 96% 的用户不受截断影响 (< 170 items)
- **Item 维度**: 被截的 4% 重度用户贡献了 ~50% 的交互，截断丢失大量 tokens
- 截断保留的是**最近的** 170 items，对推荐场景近期行为更有价值，影响可控
