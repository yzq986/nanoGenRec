"""RKMeansModelWrapper — 评测用模型包装。"""

from typing import Any, Dict

import torch


class RKMeansModelWrapper:
    """Wrapper to provide consistent interface for loaded model data"""

    def __init__(self, model_data: Dict[str, Any], device: str = 'cuda'):
        self.centroids_list = model_data['centroids_list']
        self.normalize_residuals = model_data.get('normalize_residuals', True)
        self.n_layers = model_data['n_layers']
        self.n_clusters = model_data['n_clusters']
        self.n_features = model_data.get('n_features', self.centroids_list[0].shape[1])

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.primary_device = self.device

        # Create KMeans-like layer objects
        self.kmeans_layers = [
            _KMeansLayer(centroids.to(self.device))
            for centroids in self.centroids_list
        ]


class _KMeansLayer:
    """Simple wrapper for centroids to match KMeans interface"""

    def __init__(self, centroids: torch.Tensor):
        self.centroids = centroids

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(data, self.centroids, p=2) ** 2
        return distances.argmin(dim=1)
