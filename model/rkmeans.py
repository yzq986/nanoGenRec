"""FaissKMeansLayer + ResidualQuantizationMultiGPU."""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F


class FaissKMeansLayer:
    """KMeans layer using FAISS — full-batch Lloyd's with GPU support.

    Advantages over mini-batch:
    - Full-batch Lloyd's: assign all → recompute means, guaranteed monotone descent
    - Empty cluster rebalance: auto-splits large clusters to fill empty ones
    - GPU accelerated via faiss-gpu
    """

    def __init__(self, n_clusters: int, n_features: int, gpu: bool = True):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.gpu = gpu
        self.centroids = None  # torch.Tensor on cuda

    def train(self, data: torch.Tensor, niter: int = 25, nredo: int = 1, verbose: bool = True):
        """Train KMeans on data using FAISS.

        Args:
            data: (N, D) torch tensor
            niter: number of Lloyd's iterations
            nredo: number of restarts, keep best
            verbose: print progress
        """
        import faiss
        import gc

        # faiss.Kmeans.train() only accepts numpy — required CPU copy, not avoidable
        data_np = data.cpu().numpy().astype(np.float32)

        use_gpu = self.gpu
        if use_gpu and faiss.get_num_gpus() == 0:
            raise RuntimeError(
                "[rkmeans] gpu=True but faiss.get_num_gpus()==0. "
                "Check LD_LIBRARY_PATH includes /usr/local/nvidia/lib64."
            )
        kmeans = faiss.Kmeans(
            self.n_features,
            self.n_clusters,
            niter=niter,
            nredo=nredo,
            verbose=verbose,
            gpu=use_gpu,
            seed=42,
        )
        kmeans.train(data_np)

        self.centroids = torch.tensor(kmeans.centroids, dtype=torch.float32)
        if torch.cuda.is_available():
            self.centroids = self.centroids.cuda()

        # Compute cluster assignment stats
        _, assignments = kmeans.index.search(data_np, 1)
        assignments = assignments.squeeze(1)
        cluster_counts = np.bincount(assignments, minlength=self.n_clusters)

        n_used = (cluster_counts > 0).sum()
        utilization = n_used / self.n_clusters
        inertia = kmeans.obj[-1] if len(kmeans.obj) > 0 else float('nan')

        print(f"  Final: Inertia: {inertia:.6f}")
        print(f"  Final: Utilization: {utilization:.1%} ({n_used}/{self.n_clusters})")
        print(f"  Final: Cluster counts - min: {cluster_counts.min()}, "
              f"max: {cluster_counts.max()}, mean: {cluster_counts.mean():.1f}")

        # Free FAISS GPU resources
        del kmeans
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, data: torch.Tensor, chunk_size: int = 50000) -> torch.Tensor:
        """Assign data to nearest centroid. Returns indices tensor."""
        if len(data) <= chunk_size:
            data_gpu = data.to(self.centroids.device)
            distances = torch.cdist(data_gpu, self.centroids, p=2) ** 2
            return distances.argmin(dim=1)

        # Chunk to avoid OOM on large datasets
        all_assignments = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size].to(self.centroids.device)
            distances = torch.cdist(chunk, self.centroids, p=2) ** 2
            all_assignments.append(distances.argmin(dim=1))
        return torch.cat(all_assignments, dim=0)

    def get_centroids(self) -> torch.Tensor:
        return self.centroids


class ResidualQuantizationMultiGPU:
    """RKMeans with FAISS GPU KMeans"""

    def __init__(
        self,
        n_layers: int,
        n_clusters: int,
        n_features: int,
        normalize_residuals: bool = True,
        num_gpus: int = 1,
        **kwargs,  # ignore legacy params (lr, etc.)
    ):
        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.normalize_residuals = normalize_residuals
        self.num_gpus = num_gpus
        self.primary_device = "cuda:0" if num_gpus > 0 else "cpu"
        self.gpu = num_gpus > 0

        self.kmeans_layers: List[FaissKMeansLayer] = [
            FaissKMeansLayer(n_clusters, n_features, gpu=self.gpu)
            for _ in range(n_layers)
        ]

    def train(
        self,
        embeddings: torch.Tensor,
        niter: int = 25,
        nredo: int = 1,
        **kwargs,  # ignore legacy params
    ):
        n_samples = embeddings.shape[0]
        print(f"Training RKMeans on {n_samples:,} samples (FAISS {'GPU' if self.gpu else 'CPU'})")
        print(f"Config: {self.n_layers} layers x {self.n_clusters} clusters, niter={niter}, nredo={nredo}")

        # Move to GPU once; all subsequent ops stay on device
        current_residuals = embeddings.to(self.primary_device)

        if self.normalize_residuals:
            print("  Normalizing input embeddings (layer 0 only)...")
            current_residuals = F.normalize(current_residuals, p=2, dim=1)

        for layer_idx, kmeans in enumerate(self.kmeans_layers):
            print(f"\n{'='*60}")
            print(f"Training Layer {layer_idx + 1}/{self.n_layers}")
            print(f"{'='*60}")

            # faiss.Kmeans.train requires numpy — unavoidable CPU copy here
            kmeans.train(current_residuals, niter=niter, nredo=nredo)

            # Compute residuals fully on GPU
            print("  Computing residuals for next layer...")
            assignments = kmeans.predict(current_residuals)  # GPU tensor
            current_residuals = current_residuals - kmeans.centroids[assignments]

            residual_norm = current_residuals.norm(dim=1).mean().item()
            print(f"  Residual norm (mean): {residual_norm:.6f}")

        # Final GPU cleanup after all layers trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")

    def get_centroids_list(self) -> List[torch.Tensor]:
        return [kmeans.get_centroids().cpu() for kmeans in self.kmeans_layers]

    def save(self, path: str):
        model_data = {
            'centroids_list': self.get_centroids_list(),
            'normalize_residuals': self.normalize_residuals,  # True = normalize input only (layer 0)
            'n_layers': self.n_layers,
            'n_clusters': self.n_clusters,
            'n_features': self.n_features,
            'embedding_dim': self.n_features,
        }
        torch.save(model_data, path)
        print(f"Model saved to {path}")
