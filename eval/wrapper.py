"""Model wrappers for evaluation — RKMeans and ResKmeansFSQ."""

from typing import Any, Dict

import torch

from gr_demo.model.fsq import FSQLayer, fsq_layer_from_state


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


class _FSQLayerShim:
    """Wraps FSQLayer to match _KMeansLayer interface for evaluator compatibility.

    The centroids property materializes the implicit FSQ codebook (all possible
    code -> D-dim reconstructions). predict() uses the proper PCA->quantize pipeline
    instead of distance-based assignment.
    """

    def __init__(self, fsq_layer: FSQLayer, device: str = 'cpu'):
        self._fsq = fsq_layer
        self._device = device
        # Materialize centroids lazily
        self._centroids = None

    @property
    def centroids(self) -> torch.Tensor:
        if self._centroids is None:
            all_codes = torch.arange(self._fsq.codebook_size, dtype=torch.long)
            self._centroids = self._fsq.get_centroids_for_codes(all_codes).to(self._device)
        return self._centroids

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        return self._fsq.predict(data)


class ResKmeansFSQModelWrapper:
    """Wrapper for ResKmeansFSQ model data (2 KMeans + 1 FSQ)."""

    def __init__(self, model_data: Dict[str, Any], device: str = 'cuda'):
        self.normalize_residuals = model_data.get('normalize_residuals', True)
        self.n_layers = model_data['n_layers']
        self.n_features = model_data.get('n_features', model_data['centroids_list'][0].shape[1])
        # n_clusters is not uniform — kept for backward compat but not meaningful
        self.n_clusters = model_data.get('n_kmeans_clusters', model_data['centroids_list'][0].shape[0])

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.primary_device = self.device

        # 2 KMeans layers + 1 FSQ layer, all behind same interface
        centroids_list = model_data['centroids_list']
        fsq_state = model_data['fsq_state']
        fsq_layer = fsq_layer_from_state(fsq_state)

        self.kmeans_layers = [
            _KMeansLayer(centroids_list[0].to(self.device)),
            _KMeansLayer(centroids_list[1].to(self.device)),
            _FSQLayerShim(fsq_layer, device=self.device),
        ]


def load_model_wrapper(model_data: Dict[str, Any], device: str = 'cuda'):
    """Auto-detect model type and return appropriate wrapper."""
    if model_data.get('model_type') == 'rkmeans_fsq':
        return ResKmeansFSQModelWrapper(model_data, device)
    return RKMeansModelWrapper(model_data, device)
