"""
User Behavior-based Metrics

基于用户行为数据评估 embedding 和 semantic ID 的质量。

数据格式 (from export_user_behavior.py):
    uid: 用户 ID
    iid: 内容 ID (content_id)
    action_bitmap: 行为位图
        - bit 0  (1)        click
        - bit 1  (2)        like
        - bit 2  (4)        share
        - bit 3  (8)        follow
        - bit 31 (-2147483648) negative_feedback
    first_ts, last_ts, event_cnt
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseMetric, MetricResult


# Action bitmap 定义 (from export_user_behavior.py)
# - action_bitmap > 0: 正向行为 (click, like, share, follow, etc.)
# - action_bitmap < 0: 负反馈 (negative_feedback, bit 31 符号位)


def is_negative(action: int) -> bool:
    """判断是否为负反馈 (action_bitmap < 0)"""
    return action < 0


def is_positive(action: int) -> bool:
    """判断是否为正向行为 (action_bitmap > 0)"""
    return action > 0


class UserSemanticConsistencyMetric(BaseMetric):
    """User Semantic Consistency: 同一用户正向交互的内容，semantic ID 相似度

    假设: 如果 semantic ID 有意义，同一用户喜欢的内容应该有相似的 semantic ID

    计算方式:
    1. 对每个用户，找出其正向交互的所有内容
    2. 计算这些内容的 semantic ID 的 Jaccard 相似度 (逐层 token 匹配)
    3. 对比随机 baseline
    """

    name = 'user_semantic_consistency'
    requires_model = False
    requires_semantic_ids = True

    # Higher is better
    thresholds = {
        'excellent': 0.3,  # 用户内相似度比随机高 30%+
        'good': 0.2,
        'acceptable': 0.1,
    }

    def assess_quality(self, value: float) -> str:
        """Higher lift over random is better"""
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
        min_positive_items: int = 3,
        max_users: int = 10000,
        **kwargs
    ) -> MetricResult:
        """计算用户语义一致性

        Args:
            behavior_data: Dict with 'uid', 'iid', 'action_bitmap' arrays
            content_id_to_idx: content_id -> index in semantic_ids
            min_positive_items: 用户至少要有这么多正向交互才纳入计算
            max_users: 最多采样多少用户
        """
        self.validate_inputs(embeddings, model, semantic_ids)

        if behavior_data is None or content_id_to_idx is None:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'behavior_data or content_id_to_idx not provided'},
                status='unknown',
            )

        uids = behavior_data['uid']
        iids = behavior_data['iid']
        actions = behavior_data['action_bitmap']

        # 按用户聚合正向交互的内容
        user_positive_items = defaultdict(list)
        for uid, iid, action in zip(uids, iids, actions):
            # 正向行为: like, share, follow (action > 0 排除 negative)
            if is_positive(action):
                if iid in content_id_to_idx:
                    user_positive_items[uid].append(content_id_to_idx[iid])

        # 筛选有足够正向交互的用户
        valid_users = [
            (uid, items) for uid, items in user_positive_items.items()
            if len(items) >= min_positive_items
        ]

        if not valid_users:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': f'No users with >= {min_positive_items} positive items'},
                status='unknown',
            )

        # 采样用户
        if len(valid_users) > max_users:
            import random
            random.seed(42)
            valid_users = random.sample(valid_users, max_users)

        # 计算用户内 semantic ID 相似度
        user_similarities = []
        for uid, item_indices in valid_users:
            sids = [semantic_ids[idx] for idx in item_indices]
            sim = self._compute_sid_similarity(sids)
            user_similarities.append(sim)

        user_mean_sim = np.mean(user_similarities)

        # 计算随机 baseline
        random_similarities = []
        all_sids = list(semantic_ids)
        import random
        random.seed(42)
        for _ in range(len(valid_users)):
            # 随机采样相同数量的内容
            n_items = len(valid_users[0][1])  # 使用第一个用户的 item 数
            sampled_sids = random.sample(all_sids, min(n_items, len(all_sids)))
            sim = self._compute_sid_similarity(sampled_sids)
            random_similarities.append(sim)

        random_mean_sim = np.mean(random_similarities)

        # Lift over random
        lift = (user_mean_sim - random_mean_sim) / (random_mean_sim + 1e-8)

        status = self.assess_quality(lift)

        return MetricResult(
            name=self.name,
            value=lift,
            layer_values=[],
            details={
                'user_mean_similarity': user_mean_sim,
                'random_mean_similarity': random_mean_sim,
                'lift_over_random': lift,
                'n_valid_users': len(valid_users),
                'min_positive_items': min_positive_items,
            },
            status=status,
        )

    def _compute_sid_similarity(self, sids: List[str]) -> float:
        """计算一组 semantic ID 的平均 Jaccard 相似度"""
        if len(sids) < 2:
            return 0.0

        # 解析 semantic ID: "12_34_56" -> [12, 34, 56]
        parsed = [tuple(sid.split('_')) for sid in sids]

        # 计算 pairwise Jaccard similarity (token 级别)
        similarities = []
        n = len(parsed)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._jaccard_similarity(parsed[i], parsed[j])
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def _jaccard_similarity(self, a: tuple, b: tuple) -> float:
        """计算两个 token 序列的 Jaccard 相似度"""
        # 每层 token 相同则得分
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / len(a)


class SemanticNeighborHitRateMetric(BaseMetric):
    """Semantic Neighbor Hit Rate: semantic ID 相近的内容，用户行为是否相似

    假设: 如果 semantic ID 有效，相同/相近 semantic ID 的内容应该被相同用户群喜欢

    计算方式:
    1. 对每个内容，找到 semantic ID 相同 (或前缀相同) 的邻居
    2. 计算喜欢该内容的用户，也喜欢邻居内容的比例 (命中率)
    """

    name = 'semantic_neighbor_hit_rate'
    requires_model = False
    requires_semantic_ids = True

    # Higher is better
    thresholds = {
        'excellent': 0.15,
        'good': 0.10,
        'acceptable': 0.05,
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
        content_ids: Optional[np.ndarray] = None,
        prefix_layers: int = 2,  # 使用前 N 层作为 "相近" 的定义
        max_items: int = 5000,
        **kwargs
    ) -> MetricResult:
        """计算邻居命中率"""
        self.validate_inputs(embeddings, model, semantic_ids)

        if behavior_data is None or content_id_to_idx is None or content_ids is None:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'Missing required data'},
                status='unknown',
            )

        uids = behavior_data['uid']
        iids = behavior_data['iid']
        actions = behavior_data['action_bitmap']

        # 构建 content -> users who liked it
        content_positive_users = defaultdict(set)
        for uid, iid, action in zip(uids, iids, actions):
            if is_positive(action):
                content_positive_users[iid].add(uid)

        # 构建 prefix -> content_ids
        prefix_to_contents = defaultdict(list)
        idx_to_content_id = {v: k for k, v in content_id_to_idx.items()}

        for idx, sid in enumerate(semantic_ids):
            if idx in idx_to_content_id:
                prefix = '_'.join(sid.split('_')[:prefix_layers])
                prefix_to_contents[prefix].append(idx_to_content_id[idx])

        # 采样内容计算命中率
        valid_contents = [
            cid for cid in content_positive_users.keys()
            if cid in content_id_to_idx and len(content_positive_users[cid]) >= 2
        ]

        if not valid_contents:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'No valid contents with positive users'},
                status='unknown',
            )

        import random
        random.seed(42)
        if len(valid_contents) > max_items:
            valid_contents = random.sample(valid_contents, max_items)

        # 计算命中率
        n_total_evaluated = len(valid_contents)
        hit_rates = []
        for cid in valid_contents:
            idx = content_id_to_idx[cid]
            sid = semantic_ids[idx]
            prefix = '_'.join(sid.split('_')[:prefix_layers])

            # 找邻居 (相同前缀，不含自己)
            neighbors = [c for c in prefix_to_contents[prefix] if c != cid]
            if not neighbors:
                continue

            # 喜欢当前内容的用户
            positive_users = content_positive_users[cid]

            # 命中率: 这些用户中，也喜欢邻居内容的比例
            hits = 0
            for neighbor in neighbors:
                neighbor_users = content_positive_users.get(neighbor, set())
                if positive_users & neighbor_users:  # 有交集
                    hits += 1

            hit_rate = hits / len(neighbors)
            hit_rates.append(hit_rate)

        if not hit_rates:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'No valid neighbor pairs'},
                status='unknown',
            )

        mean_hit_rate = np.mean(hit_rates)
        status = self.assess_quality(mean_hit_rate)

        return MetricResult(
            name=self.name,
            value=mean_hit_rate,
            layer_values=[],
            details={
                'mean_hit_rate': mean_hit_rate,
                'std_hit_rate': np.std(hit_rates),
                'n_contents_evaluated': len(hit_rates),
                'n_total_sampled': n_total_evaluated,
                'neighbor_coverage': len(hit_rates) / n_total_evaluated if n_total_evaluated > 0 else 0.0,
                'prefix_layers': prefix_layers,
            },
            status=status,
        )


class EmbeddingBehaviorCorrelationMetric(BaseMetric):
    """Embedding-Behavior Correlation: embedding 相似度与用户行为相似度的相关性

    假设: 如果 embedding 有意义，相似的内容应该被相似的用户群喜欢

    计算方式:
    1. 采样内容对
    2. 计算 embedding cosine similarity
    3. 计算用户重叠 (Jaccard)
    4. 计算两者的 Spearman 相关系数
    """

    name = 'embedding_behavior_correlation'
    requires_model = False
    requires_semantic_ids = False

    # Higher correlation is better
    thresholds = {
        'excellent': 0.3,
        'good': 0.2,
        'acceptable': 0.1,
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
        n_pairs: int = 10000,
        **kwargs
    ) -> MetricResult:
        """计算 embedding-behavior 相关性"""
        self.validate_inputs(embeddings, model, semantic_ids)

        if behavior_data is None or content_id_to_idx is None:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'Missing required data'},
                status='unknown',
            )

        uids = behavior_data['uid']
        iids = behavior_data['iid']
        actions = behavior_data['action_bitmap']

        # 构建 content -> positive users
        content_users = defaultdict(set)
        for uid, iid, action in zip(uids, iids, actions):
            if is_positive(action):  # action > 0
                if iid in content_id_to_idx:
                    content_users[iid].add(uid)

        # 筛选有足够用户的内容
        valid_contents = [
            cid for cid, users in content_users.items()
            if len(users) >= 5 and cid in content_id_to_idx
        ]

        if len(valid_contents) < 100:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': f'Not enough valid contents: {len(valid_contents)}'},
                status='unknown',
            )

        # 随机采样内容对
        import random
        random.seed(42)
        pairs = []
        for _ in range(n_pairs):
            c1, c2 = random.sample(valid_contents, 2)
            pairs.append((c1, c2))

        # 计算 embedding similarity 和 user overlap
        emb_similarities = []
        user_overlaps = []

        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)

        for c1, c2 in pairs:
            idx1 = content_id_to_idx[c1]
            idx2 = content_id_to_idx[c2]

            # Embedding cosine similarity
            emb_sim = (embeddings_norm[idx1] @ embeddings_norm[idx2]).item()
            emb_similarities.append(emb_sim)

            # User Jaccard overlap
            users1 = content_users[c1]
            users2 = content_users[c2]
            intersection = len(users1 & users2)
            union = len(users1 | users2)
            jaccard = intersection / union if union > 0 else 0
            user_overlaps.append(jaccard)

        # Spearman correlation
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(emb_similarities, user_overlaps)

        status = self.assess_quality(correlation)

        return MetricResult(
            name=self.name,
            value=correlation,
            layer_values=[],
            details={
                'spearman_correlation': correlation,
                'p_value': p_value,
                'n_pairs': len(pairs),
                'n_valid_contents': len(valid_contents),
                'emb_sim_mean': np.mean(emb_similarities),
                'user_overlap_mean': np.mean(user_overlaps),
            },
            status=status,
        )


class PositiveNegativeSeparationMetric(BaseMetric):
    """Positive-Negative Separation: 正负样本在 embedding 空间的分离度

    假设: 好的 embedding 应该让用户喜欢的内容彼此接近，不喜欢的远离

    计算方式:
    1. 对每个用户，找出正向交互和负向交互的内容
    2. 计算 pos-pos 距离 vs pos-neg 距离
    3. 计算分离度 = (neg_dist - pos_dist) / neg_dist
    """

    name = 'positive_negative_separation'
    requires_model = False
    requires_semantic_ids = False

    # Higher separation is better
    thresholds = {
        'excellent': 0.15,
        'good': 0.10,
        'acceptable': 0.05,
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
        min_items: int = 2,
        max_users: int = 5000,
        **kwargs
    ) -> MetricResult:
        """计算正负样本分离度"""
        self.validate_inputs(embeddings, model, semantic_ids)

        if behavior_data is None or content_id_to_idx is None:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'Missing required data'},
                status='unknown',
            )

        uids = behavior_data['uid']
        iids = behavior_data['iid']
        actions = behavior_data['action_bitmap']

        # 按用户聚合正负交互
        user_positive = defaultdict(list)
        user_negative = defaultdict(list)

        for uid, iid, action in zip(uids, iids, actions):
            if iid not in content_id_to_idx:
                continue

            idx = content_id_to_idx[iid]
            if is_negative(action):  # action < 0 表示 negative feedback
                user_negative[uid].append(idx)
            elif is_positive(action):  # action > 0 且有 like/share/follow
                user_positive[uid].append(idx)

        # 筛选有正负样本的用户
        valid_users = [
            uid for uid in user_positive
            if len(user_positive[uid]) >= min_items and len(user_negative[uid]) >= min_items
        ]

        if not valid_users:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'No users with both positive and negative samples'},
                status='unknown',
            )

        import random
        random.seed(42)
        if len(valid_users) > max_users:
            valid_users = random.sample(valid_users, max_users)

        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)

        # 计算分离度
        separations = []
        for uid in valid_users:
            pos_indices = user_positive[uid]
            neg_indices = user_negative[uid]

            pos_embs = embeddings_norm[pos_indices]
            neg_embs = embeddings_norm[neg_indices]

            # pos-pos 距离 (1 - cosine)
            if len(pos_indices) >= 2:
                pos_sim = (pos_embs @ pos_embs.t()).mean().item()
                pos_dist = 1 - pos_sim
            else:
                pos_dist = 0

            # pos-neg 距离
            pos_neg_sim = (pos_embs @ neg_embs.t()).mean().item()
            neg_dist = 1 - pos_neg_sim

            # 分离度
            if neg_dist > 0:
                separation = (neg_dist - pos_dist) / neg_dist
                separations.append(separation)

        if not separations:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'Could not compute separations'},
                status='unknown',
            )

        mean_separation = np.mean(separations)
        status = self.assess_quality(mean_separation)

        return MetricResult(
            name=self.name,
            value=mean_separation,
            layer_values=[],
            details={
                'mean_separation': mean_separation,
                'std_separation': np.std(separations),
                'n_users': len(separations),
            },
            status=status,
        )
