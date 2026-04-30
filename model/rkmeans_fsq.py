"""ResKmeansFSQ — 2 layers RKMeans + 1 layer FSQ.

Hybrid quantizer: first two layers use FAISS KMeans (residual quantization),
third layer uses Finite Scalar Quantization (PCA + per-dim rounding).

Reference: OneMall (arxiv 2601.21770) — FSQ on residuals reduces collision.
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from model.rkmeans import FaissKMeansLayer
from model.fsq import FSQLayer, LearnedFSQLayer, fsq_layer_from_state


class ResKmeansFSQ:
    """2-layer RKMeans + 1-layer FSQ hybrid quantizer."""

    def __init__(
        self,
        n_kmeans_clusters,  # int (same for both layers) or List[int] (per-layer)
        fsq_levels: List[int],
        n_features: int,
        normalize_residuals: bool = True,
        num_gpus: int = 1,
        fsq_projection: str = 'pca',
        fsq_mlp_hidden: int = 128,
        fsq_epochs: int = 50,
    ):
        # Accept int, "4096", or "4096,2048" — normalise to List[int]
        if isinstance(n_kmeans_clusters, str):
            n_kmeans_clusters = [int(x) for x in n_kmeans_clusters.split(",")]
        if isinstance(n_kmeans_clusters, int):
            n_kmeans_clusters = [n_kmeans_clusters, n_kmeans_clusters]
        self.n_kmeans_clusters = n_kmeans_clusters  # List[int], one per KMeans layer
        self.fsq_levels = fsq_levels
        self.n_features = n_features
        self.normalize_residuals = normalize_residuals
        self.num_gpus = num_gpus
        self.n_layers = 3  # always: 2 KMeans + 1 FSQ
        self.primary_device = "cuda:0" if num_gpus > 0 else "cpu"
        self.gpu = num_gpus > 0

        self.kmeans_layers: List[FaissKMeansLayer] = [
            FaissKMeansLayer(nc, n_features, gpu=self.gpu)
            for nc in self.n_kmeans_clusters
        ]
        if fsq_projection == 'mlp':
            self.fsq_layer = LearnedFSQLayer(
                fsq_levels, n_features,
                hidden_dim=fsq_mlp_hidden,
                epochs=fsq_epochs,
                device=self.primary_device,
            )
        else:
            self.fsq_layer = FSQLayer(fsq_levels, n_features)

    def train(
        self,
        embeddings: torch.Tensor,
        niter: int = 25,
        nredo: int = 1,
    ):
        n_samples = embeddings.shape[0]
        print(f"Training ResKmeansFSQ on {n_samples:,} samples")
        nc_str = "×".join(str(n) for n in self.n_kmeans_clusters)
        print(f"Config: 2 KMeans ({nc_str} clusters) + 1 FSQ ({self.fsq_levels})")

        current_residuals = embeddings.clone()

        # Normalize input once (layer 0 only)
        if self.normalize_residuals:
            print("  Normalizing input embeddings (layer 0 only)...")
            normalized = []
            chunk_size = 100000
            for i in range(0, n_samples, chunk_size):
                chunk = current_residuals[i:i+chunk_size].to(self.primary_device)
                chunk = F.normalize(chunk, p=2, dim=1).cpu()
                normalized.append(chunk)
            current_residuals = torch.cat(normalized, dim=0)

        # Layer 1 & 2: KMeans
        for layer_idx, kmeans in enumerate(self.kmeans_layers):
            print(f"\n{'='*60}")
            print(f"Training KMeans Layer {layer_idx + 1}/2")
            print(f"{'='*60}")

            kmeans.train(current_residuals, niter=niter, nredo=nredo)

            # Compute residuals for next layer
            print("  Computing residuals for next layer...")
            new_residuals = []
            chunk_size = 50000
            for i in range(0, n_samples, chunk_size):
                chunk = current_residuals[i:i+chunk_size]
                assignments = kmeans.predict(chunk)
                assigned_centroids = kmeans.centroids[assignments].cpu()
                residual = chunk - assigned_centroids
                new_residuals.append(residual)
            current_residuals = torch.cat(new_residuals, dim=0)

            residual_norm = torch.norm(current_residuals, dim=1).mean().item()
            print(f"  Residual norm (mean): {residual_norm:.6f}")

        # Layer 3: FSQ
        print(f"\n{'='*60}")
        print(f"Training FSQ Layer 3/3")
        print(f"{'='*60}")

        self.fsq_layer.train(current_residuals)

        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'ResKmeansFSQ':
        """Load a trained quantizer from .pt file (predict-only, no retrain)."""
        model_data = torch.load(path, map_location=device, weights_only=False)

        obj = cls.__new__(cls)
        obj.normalize_residuals = model_data['normalize_residuals']
        obj.n_layers = model_data['n_layers']
        obj.fsq_levels = model_data['fsq_levels']
        obj.n_features = model_data['n_features']
        obj.num_gpus = 0
        obj.primary_device = device
        obj.gpu = False

        # Rebuild KMeans layers from saved centroids
        obj.kmeans_layers = []
        for centroids in model_data['centroids_list']:
            layer = FaissKMeansLayer(centroids.shape[0], centroids.shape[1], gpu=False)
            layer.centroids = centroids.to(device)
            obj.kmeans_layers.append(layer)
        # Restore per-layer cluster counts (old checkpoints stored a scalar)
        saved_nc = model_data['n_kmeans_clusters']
        if isinstance(saved_nc, int):
            saved_nc = [saved_nc] * len(obj.kmeans_layers)
        obj.n_kmeans_clusters = saved_nc

        # Rebuild FSQ layer from saved state
        obj.fsq_layer = fsq_layer_from_state(model_data['fsq_state'])

        nc_str = "×".join(str(n) for n in obj.n_kmeans_clusters)
        print(f"Loaded quantizer from {path} "
              f"(KMeans [{nc_str}] + FSQ {obj.fsq_levels})")
        return obj

    def save(self, path: str):
        model_data = {
            'model_type': 'rkmeans_fsq',
            'centroids_list': [km.get_centroids().cpu() for km in self.kmeans_layers],
            'fsq_state': self.fsq_layer.save_state(),
            'normalize_residuals': self.normalize_residuals,
            'n_layers': self.n_layers,
            'n_kmeans_clusters': self.n_kmeans_clusters,
            'fsq_levels': self.fsq_levels,
            'n_features': self.n_features,
            'embedding_dim': self.n_features,
        }
        torch.save(model_data, path)
        print(f"Model saved to {path}")


def generate_semantic_ids_fsq(
    model: ResKmeansFSQ,
    embeddings: torch.Tensor,
    normalize_residuals: bool = True,
) -> List[str]:
    """Generate semantic IDs: 2 KMeans layers + 1 FSQ layer -> "c1_c2_c3" strings."""
    n_samples = embeddings.shape[0]
    device = model.primary_device

    layer_assignments = []
    # Keep residuals on GPU throughout to avoid CPU bottleneck
    current_residuals = embeddings.clone().to(device)

    # Normalize input once (layer 0 only)
    if normalize_residuals:
        chunk_size = 100000
        normalized = []
        for i in range(0, n_samples, chunk_size):
            chunk = F.normalize(current_residuals[i:i+chunk_size], p=2, dim=1)
            normalized.append(chunk)
        current_residuals = torch.cat(normalized, dim=0)

    # Layers 1 & 2: KMeans
    for layer_idx, kmeans in enumerate(model.kmeans_layers):
        print(f"  Predicting KMeans layer {layer_idx + 1}/2...")
        assignments = kmeans.predict(current_residuals).cpu().numpy()
        layer_assignments.append(assignments)

        # Compute residuals on GPU
        chunk_size = 50000
        new_residuals = []
        for i in range(0, n_samples, chunk_size):
            chunk = current_residuals[i:i+chunk_size]
            chunk_assignments = torch.tensor(assignments[i:i+chunk_size], device=device)
            assigned_centroids = kmeans.centroids[chunk_assignments]
            new_residuals.append(chunk - assigned_centroids)
        current_residuals = torch.cat(new_residuals, dim=0)

    # Layer 3: FSQ
    print(f"  Predicting FSQ layer 3/3...")
    fsq_codes = model.fsq_layer.predict(current_residuals).cpu().numpy()
    layer_assignments.append(fsq_codes)

    # Build SID strings
    semantic_ids = []
    for i in range(n_samples):
        sid = "_".join(str(layer_assignments[layer][i]) for layer in range(3))
        semantic_ids.append(sid)

    return semantic_ids
