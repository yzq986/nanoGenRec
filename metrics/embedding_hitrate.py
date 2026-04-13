"""
Embedding Hit Rate (FORGE proxy metric)

用 item embedding 做 I2I top-K 检索，计算检索邻居在用户行为数据中的共现率。
衡量 "embedding 空间的相似性是否反映了真实用户行为"。

该指标与下游 GR 推荐性能正相关 (FORGE, arxiv 2509.20904)，
可以在不训练 NTP 模型的情况下快速评估 embedding / tokenizer 质量。

公式:
    HR@K = (1/N) * Σ |I_K(i) ∩ I_click(i)| / |I_click(i)|

其中 I_K(i) 是 embedding 空间中 item i 的 top-K 邻居,
I_click(i) 是与 item i 被同一用户正向交互过的 item 集合。
"""

from typing import Any, Dict, List, Optional
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseMetric, MetricResult


class EmbeddingHitRateMetric(BaseMetric):
    """Embedding Hit Rate: embedding 近邻与行为共现的重合度

    FORGE (arxiv 2509.20904) 提出的 SID proxy metric。
    无需训练 GR 模型即可评估 embedding 质量。
    """

    name = 'embedding_hit_rate'
    requires_model = False
    requires_semantic_ids = False

    # Higher is better
    thresholds = {
        'excellent': 0.10,
        'good': 0.05,
        'acceptable': 0.02,
    }

    def assess_quality(self, value: float) -> str:
        if value >= self.thresholds['excellent']:
            return 'excellent'
        elif value >= self.thresholds['good']:
            return 'good'
        elif value >= self.thresholds['acceptable']:
            return 'acceptable'
        else:
            return 'poor'

    def compute(
        self,
        embeddings: torch.Tensor,
        model: Optional[Any] = None,
        semantic_ids: Optional[List[str]] = None,
        layer_assignments: Optional[List[torch.Tensor]] = None,
        behavior_data: Optional[Dict] = None,
        content_id_to_idx: Optional[Dict[str, int]] = None,
        top_k_list: Optional[List[int]] = None,
        max_items: int = 50000,
        **kwargs
    ) -> MetricResult:
        """计算 Embedding Hit Rate

        Args:
            embeddings: (N, D) item embeddings
            behavior_data: Dict with 'uid', 'iid', 'action_bitmap' arrays
            content_id_to_idx: content_id -> index in embeddings
            top_k_list: K values for HR@K, default [10, 50, 100, 500]
            max_items: 最多评估多少 item (采样)
        """
        self.validate_inputs(embeddings, model, semantic_ids)

        if behavior_data is None or content_id_to_idx is None:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'behavior_data or content_id_to_idx not provided'},
                status='unknown',
            )

        if top_k_list is None:
            top_k_list = [10, 50, 100, 500]

        # --- Step 1: Build item co-occurrence from user behavior ---
        uids = behavior_data['uid']
        iids = behavior_data['iid']
        actions = behavior_data['action_bitmap']

        # user -> set of positively interacted item indices
        user_items: Dict[Any, set] = defaultdict(set)
        # item index -> set of users
        item_users: Dict[int, set] = defaultdict(set)

        for uid, iid, action in zip(uids, iids, actions):
            if action > 0 and iid in content_id_to_idx:
                idx = content_id_to_idx[iid]
                user_items[uid].add(idx)
                item_users[idx].add(uid)

        # item index -> co-occurrence items (items liked by the same users)
        # For efficiency, only compute for sampled items
        valid_indices = [idx for idx in item_users if len(item_users[idx]) >= 1]

        if not valid_indices:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'No items with positive interactions'},
                status='unknown',
            )

        # Sample items for evaluation
        import random
        random.seed(42)
        if len(valid_indices) > max_items:
            eval_indices = random.sample(valid_indices, max_items)
        else:
            eval_indices = valid_indices

        # Build co-occurrence set for each eval item
        item_cooccurrence: Dict[int, set] = {}
        for idx in eval_indices:
            cooccur = set()
            for uid in item_users[idx]:
                cooccur.update(user_items[uid])
            cooccur.discard(idx)  # remove self
            if cooccur:
                item_cooccurrence[idx] = cooccur

        eval_indices = [idx for idx in eval_indices if idx in item_cooccurrence]

        if not eval_indices:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'No items with co-occurrence data'},
                status='unknown',
            )

        # --- Step 2: FAISS top-K retrieval ---
        max_k = max(top_k_list)

        embeddings_np = embeddings.float().cpu().numpy()
        # L2 normalize for inner product = cosine similarity
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings_np = embeddings_np / norms

        import faiss
        d = embeddings_np.shape[1]
        n = embeddings_np.shape[0]

        # Use GPU if available, otherwise CPU
        if n <= 500_000:
            # Brute-force is fine for moderate scale
            index = faiss.IndexFlatIP(d)
            index.add(embeddings_np.astype(np.float32))
        else:
            # IVF for larger scale
            nlist = min(4096, n // 100)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings_np.astype(np.float32))
            index.add(embeddings_np.astype(np.float32))
            index.nprobe = min(64, nlist)

        # Query in batches
        query_embeddings = embeddings_np[eval_indices].astype(np.float32)
        batch_size = 4096
        all_neighbors = np.zeros((len(eval_indices), max_k + 1), dtype=np.int64)

        for start in range(0, len(eval_indices), batch_size):
            end = min(start + batch_size, len(eval_indices))
            batch = query_embeddings[start:end]
            _, I = index.search(batch, max_k + 1)  # +1 to exclude self
            all_neighbors[start:end] = I

        # --- Step 3: Compute HR@K ---
        hr_results = {}
        for k in top_k_list:
            hit_rates = []
            for i, idx in enumerate(eval_indices):
                neighbors = all_neighbors[i]
                # Exclude self from neighbors
                topk = [n for n in neighbors if n != idx][:k]
                topk_set = set(topk)

                cooccur = item_cooccurrence[idx]
                hits = len(topk_set & cooccur)
                hr = hits / len(cooccur) if cooccur else 0.0
                hit_rates.append(hr)

            hr_results[f'HR@{k}'] = float(np.mean(hit_rates))

        # Primary value = HR@50 (FORGE default)
        primary_k = 50 if 50 in top_k_list else top_k_list[0]
        primary_value = hr_results[f'HR@{primary_k}']

        status = self.assess_quality(primary_value)

        details = {
            **hr_results,
            'n_items_evaluated': len(eval_indices),
            'n_items_with_behavior': len(valid_indices),
            'primary_metric': f'HR@{primary_k}',
        }

        return MetricResult(
            name=self.name,
            value=primary_value,
            layer_values=[],
            details=details,
            status=status,
        )
