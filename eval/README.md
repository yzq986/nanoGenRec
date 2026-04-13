# eval/ — 评测框架

单模型评测、多模型对比、批量评测编排、超参数搜索。

## 文件说明

| 文件 | 说明 |
|------|------|
| `wrapper.py` | RKMeansModelWrapper — 评测用模型包装 |
| `evaluator.py` | MetricsEvaluator — 指标注册与运行核心 |
| `behavior.py` | BehaviorMetricsEvaluator — 扩展行为指标支持 |
| `compare.py` | 多模型对比报告生成 (Markdown / JSON / CSV) |
| `batch.py` | 批量评测编排 — 一键跑全部模型 |
| `hyperparam.py` | 超参数网格搜索 (num_clusters, niter, nredo) |

## 评测流程

```
wrapper.py (加载模型) → evaluator.py (注册+运行 intrinsic 指标)
                      → behavior.py (+ 行为指标)
                      → batch.py (多模型批量)
                      → compare.py (对比报告)
```

## CLI 用法

```bash
# 单模型评测
python -m gr_demo eval --results_path s3://... --model_path s3://...

# 批量评测
python -m gr_demo eval-all --models qwen3-0.6b qwen3-4b --quick

# 仅对比已有结果
python -m gr_demo compare --eval_dir eval_results

# 超参搜索 (默认跑 intrinsic + embedding_hit_rate)
python -m gr_demo hyperparam --model qwen3-0.6b --skip_embedding

# 超参搜索 + NTP (慢，需要显式开启)
python -m gr_demo hyperparam --model qwen3-0.6b --skip_embedding --run_ntp
```

## evaluator.py

核心评测器，预计算逐层 cluster assignment 缓存，避免每个 metric 重复计算：

- `register_intrinsic_metrics()` — 注册全部 intrinsic 指标
- `register_metrics(names)` — 按名称选择性注册
- `evaluate(metric_kwargs)` — 运行全部已注册指标

## behavior.py

扩展 `MetricsEvaluator`，增加行为数据上下文：

- 加载用户行为数据 (uid, iid, action_bitmap, first_ts)
- 行为指标: 语义一致性、邻居命中率、embedding-行为相关性、正负样本分离度
- Embedding 命中率 (FORGE proxy): FAISS I2I 检索邻居与行为共现率，**默认开启**
- Semantic ID 预测 (NTP): 训练 Transformer + MoE，beam search 评估。**默认关闭**，需 `--run_ntp` 开启

## hyperparam.py

网格搜索 RKMeans 超参数：

- 搜索维度: `num_clusters` (256/512/1024/2048), `niter`, `nredo`
- 支持断点续搜 (append mode)
- 生成 Markdown 报告 + top-5 最优配置
