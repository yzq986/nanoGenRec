"""generate_semantic_ids() — 每层聚类 ID 拼接成字符串。"""

from typing import List

import torch
import torch.nn.functional as F


def generate_semantic_ids(
    model: 'ResidualQuantizationMultiGPU',
    embeddings: torch.Tensor,
    normalize_residuals: bool = True
) -> List[str]:
    """生成 semantic_id: 每层聚类 ID 拼接成字符串

    例如: "12_34_56" 表示第1层聚类12, 第2层聚类34, 第3层聚类56
    """
    n_samples = embeddings.shape[0]
    device = model.primary_device

    # 收集每层的聚类 ID
    layer_assignments = []
    current_residuals = embeddings.clone()

    # 只对原始输入做 L2 normalize（layer 0），残差保留原始 scale
    if normalize_residuals:
        normalized = []
        chunk_size = 100000
        for i in range(0, n_samples, chunk_size):
            chunk = current_residuals[i:i+chunk_size].to(device)
            chunk = F.normalize(chunk, p=2, dim=1).cpu()
            normalized.append(chunk)
        current_residuals = torch.cat(normalized, dim=0)

    for layer_idx, kmeans in enumerate(model.kmeans_layers):
        print(f"  Predicting layer {layer_idx + 1}/{model.n_layers}...")

        # Predict cluster assignments
        assignments = kmeans.predict(current_residuals).cpu().numpy()
        layer_assignments.append(assignments)

        # Compute residuals for next layer
        new_residuals = []
        chunk_size = 50000
        for i in range(0, n_samples, chunk_size):
            chunk = current_residuals[i:i+chunk_size]
            chunk_assignments = torch.tensor(assignments[i:i+chunk_size])
            assigned_centroids = kmeans.centroids[chunk_assignments].cpu()
            residual = chunk - assigned_centroids
            new_residuals.append(residual)
        current_residuals = torch.cat(new_residuals, dim=0)

    # 拼接成 semantic_id 字符串
    semantic_ids = []
    for i in range(n_samples):
        sid = "_".join(str(layer_assignments[layer][i]) for layer in range(model.n_layers))
        semantic_ids.append(sid)

    return semantic_ids
