"""Model wrappers for evaluation — RKMeans, ResKmeansFSQ, and OPQ."""

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


class _OPQSubspaceLayer:
    """Wraps one OPQ subspace to match _KMeansLayer interface.

    For OPQ, each "layer" is a subspace — NOT a residual layer.
    centroids are projected back to full D-dim original space so that
    sum(centroids[assignments] for all layers) gives the full reconstruction.
    """

    def __init__(self, full_centroids: torch.Tensor, sub_codes: torch.Tensor):
        """
        Args:
            full_centroids: (M, D) centroids in original space (inverse-rotated, zero-padded)
            sub_codes: (N,) pre-computed assignments for this subspace
        """
        self.centroids = full_centroids
        self._sub_codes = sub_codes

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Not used — OPQ layer_assignments are pre-computed and set directly on evaluator.

        This exists only to satisfy the interface. Should not be called in chunked loops.
        """
        raise RuntimeError(
            "OPQ predict() should not be called. "
            "Set evaluator.layer_assignments directly from OPQ codes."
        )


class OPQModelWrapper:
    """Wrapper for OPQ model data.

    Exposes m subspace layers as kmeans_layers for evaluator compatibility.
    Key difference from RKMeans: layers are PARALLEL (not residual).
    The evaluator's _precompute_assignments does sequential residual subtraction,
    which is wrong for OPQ. So we pre-compute assignments and provide
    full-space centroids such that the sum-of-centroids reconstruction
    is mathematically equivalent to OPQ decode.
    """

    def __init__(self, model_data: Dict[str, Any], codes: 'np.ndarray', device: str = 'cuda'):
        """
        Args:
            model_data: OPQ model dict (from OPQQuantizer.save())
            codes: (N, m) pre-computed OPQ codes for eval embeddings
            device: computation device
        """
        from gr_demo.model.opq import OPQQuantizer

        self.normalize_residuals = model_data.get('normalize_input', True)
        self.n_subvectors = model_data['n_subvectors']
        self.n_layers = self.n_subvectors
        self.n_clusters = model_data['n_clusters_per_sub']
        self.n_features = model_data['n_features']

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.primary_device = self.device

        # Rebuild OPQ to get full-space centroids
        opq = OPQQuantizer.from_saved(model_data)
        fullspace_centroids = opq.get_fullspace_centroids()

        # Build subspace layers with pre-computed codes
        self.kmeans_layers = []
        for j in range(self.n_subvectors):
            sub_codes = torch.tensor(codes[:, j], dtype=torch.long)
            layer = _OPQSubspaceLayer(
                full_centroids=fullspace_centroids[j].to(self.device),
                sub_codes=sub_codes,
            )
            self.kmeans_layers.append(layer)


def load_model_wrapper(model_data: Dict[str, Any], device: str = 'cuda'):
    """Auto-detect model type and return appropriate wrapper."""
    model_type = model_data.get('model_type')
    if model_type == 'rkmeans_fsq':
        return ResKmeansFSQModelWrapper(model_data, device)
    # OPQ requires codes to be passed separately — use OPQModelWrapper directly
    return RKMeansModelWrapper(model_data, device)
