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
