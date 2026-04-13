"""OPQ (Optimized Product Quantization) for parallel semantic IDs.

Reference: Meta RPG (KDD'25, arxiv 2506.05781)
- OPQ splits embedding into m subvectors, each independently quantized
- Tokens are unordered/parallel (no residual dependency)
- FAISS OPQMatrix learns rotation R, ProductQuantizer does per-subspace KMeans
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def build_sid_graph(
    quantizer: 'OPQQuantizer',
    valid_sids: np.ndarray,
    top_k: int = 100,
    nprobe: int = 32,
    verbose: bool = True,
) -> np.ndarray:
    """Build SID similarity graph via reconstructed embeddings.

    1. Decode all valid SIDs → reconstructed vectors
    2. Build FAISS IVF index for fast ANN search
    3. Return neighbor adjacency list

    Args:
        quantizer: Trained OPQQuantizer
        valid_sids: (N, m) int array — codes for all valid SIDs
        top_k: Number of neighbors per SID
        nprobe: IVF nprobe for search quality
        verbose: Print progress

    Returns:
        neighbors: (N, top_k) int32 array — neighbor indices
    """
    import faiss
    import time

    t0 = time.time()
    N = valid_sids.shape[0]

    if verbose:
        print(f"  Building SID graph: {N:,} SIDs, top_k={top_k}")

    # 1. Reconstruct embeddings from codes
    recon = quantizer.decode_numpy(valid_sids).astype(np.float32)

    # 2. Build FAISS index (IVF for speed)
    d = recon.shape[1]
    nlist = min(int(np.sqrt(N)), 4096)

    # Use GPU if available
    try:
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatIP(d)
        index_ivf = faiss.IndexIVFFlat(index_flat, d, nlist, faiss.METRIC_INNER_PRODUCT)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_ivf)
        # Normalize for cosine similarity
        faiss.normalize_L2(recon)
        gpu_index.train(recon)
        gpu_index.add(recon)
        gpu_index.nprobe = nprobe
        _, neighbors = gpu_index.search(recon, top_k + 1)  # +1 because self is included
    except Exception:
        if verbose:
            print("    GPU FAISS not available, using CPU")
        faiss.normalize_L2(recon)
        index = faiss.IndexFlatIP(d)
        index.add(recon)
        _, neighbors = index.search(recon, top_k + 1)

    # Remove self-neighbor (first column is usually self)
    # For each row, remove the index that matches its own position
    clean_neighbors = np.zeros((N, top_k), dtype=np.int32)
    for i in range(N):
        row = neighbors[i]
        mask = row != i
        filtered = row[mask][:top_k]
        clean_neighbors[i, :len(filtered)] = filtered
        if len(filtered) < top_k:
            clean_neighbors[i, len(filtered):] = filtered[-1] if len(filtered) > 0 else 0

    elapsed = time.time() - t0
    if verbose:
        print(f"  SID graph built in {elapsed:.1f}s")

    return clean_neighbors


class OPQQuantizer:
    """OPQ quantizer using FAISS.

    Learns rotation matrix R + m independent codebooks.
    Each item gets m-token semantic ID where tokens are parallel (not sequential).
    """

    def __init__(
        self,
        n_features: int,
        n_subvectors: int,
        n_clusters_per_sub: int = 256,
        normalize_input: bool = True,
    ):
        self.n_features = n_features
        self.n_subvectors = n_subvectors
        self.n_clusters_per_sub = n_clusters_per_sub
        self.normalize_input = normalize_input
        self.sub_dim = n_features // n_subvectors

        if n_features % n_subvectors != 0:
            raise ValueError(
                f"n_features ({n_features}) must be divisible by "
                f"n_subvectors ({n_subvectors})"
            )

        # Set after training
        self.opq_matrix = None  # faiss.OPQMatrix
        self.pq = None          # faiss.ProductQuantizer
        self.index = None       # faiss.IndexPQ with OPQ pre-transform
        self._rotation = None   # (D, D) numpy rotation matrix
        self._codebooks = None  # (m, M, sub_dim) numpy codebooks

    def train(self, embeddings: torch.Tensor, verbose: bool = True):
        """Train OPQ on embeddings.

        Args:
            embeddings: (N, D) tensor
            verbose: print progress
        """
        import faiss
        import gc

        n_samples = embeddings.shape[0]
        if verbose:
            print(f"Training OPQ on {n_samples:,} samples")
            print(f"Config: {self.n_subvectors} subvectors x "
                  f"{self.n_clusters_per_sub} clusters, sub_dim={self.sub_dim}")

        data = embeddings.cpu().numpy().astype(np.float32)

        # Optionally L2 normalize input
        if self.normalize_input:
            if verbose:
                print("  Normalizing input embeddings...")
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            data = data / norms

        # Number of bits per subquantizer: log2(n_clusters_per_sub)
        nbits = int(np.log2(self.n_clusters_per_sub))
        if 2 ** nbits != self.n_clusters_per_sub:
            raise ValueError(
                f"n_clusters_per_sub ({self.n_clusters_per_sub}) must be a power of 2"
            )

        if verbose:
            print(f"  FAISS OPQ: D={self.n_features}, M={self.n_subvectors}, "
                  f"nbits={nbits}")

        # Build OPQ index: OPQ pre-transform + PQ
        opq = faiss.OPQMatrix(self.n_features, self.n_subvectors)
        pq = faiss.ProductQuantizer(self.n_features, self.n_subvectors, nbits)

        # Train OPQ rotation matrix (iterative: rotate + train PQ + re-estimate rotation)
        if verbose:
            print("  Training OPQ rotation matrix...")
        opq.pq = pq
        opq.train(data)

        # Train PQ codebooks on rotated data
        rotated_data = opq.apply(data)
        if verbose:
            print("  Training PQ codebooks on rotated data...")
        pq.train(rotated_data)

        # Store trained components
        self.opq_matrix = opq
        self.pq = pq

        # Extract rotation matrix for later use
        self._rotation = faiss.vector_to_array(opq.A).reshape(
            self.n_features, self.n_features
        )

        # Extract codebooks: (m, M, sub_dim)
        centroids = faiss.vector_to_array(pq.centroids).reshape(
            self.n_subvectors, self.n_clusters_per_sub, self.sub_dim
        )
        self._codebooks = centroids

        if verbose:
            # Compute and report stats
            codes = self.encode_numpy(data)
            recon = self.decode_numpy(codes)
            recon_loss = np.mean(np.sum((data - recon) ** 2, axis=1))
            print(f"  Reconstruction loss (MSE): {recon_loss:.6f}")

            # Per-subvector utilization
            for j in range(min(self.n_subvectors, 4)):  # show first 4
                n_unique = len(np.unique(codes[:, j]))
                print(f"  Subvector {j}: {n_unique}/{self.n_clusters_per_sub} "
                      f"codes used ({n_unique/self.n_clusters_per_sub:.1%})")
            if self.n_subvectors > 4:
                print(f"  ... ({self.n_subvectors - 4} more subvectors)")

        # Cleanup
        gc.collect()

    def encode_numpy(self, data: np.ndarray) -> np.ndarray:
        """Encode to per-subvector codes. Returns (N, m) uint8/uint16 array."""
        rotated = self.opq_matrix.apply(data)
        codes = self.pq.compute_codes(rotated)
        # FAISS returns packed bytes; decode to per-subvector indices
        return self._unpack_codes(codes)

    def decode_numpy(self, codes: np.ndarray) -> np.ndarray:
        """Decode from per-subvector codes to reconstructed embeddings."""
        # Reconstruct in rotated space
        packed = self._pack_codes(codes)
        rotated_recon = self.pq.decode(packed)
        # Apply inverse rotation (R is orthogonal, so R_inv = R.T)
        return rotated_recon @ self._rotation

    def encode(self, embeddings: torch.Tensor) -> np.ndarray:
        """Encode torch tensor to codes. Returns (N, m) numpy array."""
        data = embeddings.cpu().numpy().astype(np.float32)
        if self.normalize_input:
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            data = data / norms
        return self.encode_numpy(data)

    def decode(self, codes: np.ndarray) -> torch.Tensor:
        """Decode codes back to torch tensor."""
        recon = self.decode_numpy(codes)
        return torch.tensor(recon, dtype=torch.float32)

    def _unpack_codes(self, packed: np.ndarray) -> np.ndarray:
        """Unpack FAISS packed codes to (N, m) per-subvector indices."""
        n = packed.shape[0]
        nbits = int(np.log2(self.n_clusters_per_sub))

        if nbits == 8:
            # Each byte is one code
            return packed.reshape(n, self.n_subvectors).astype(np.int64)
        elif nbits < 8:
            # Need bit unpacking
            codes = np.zeros((n, self.n_subvectors), dtype=np.int64)
            for j in range(self.n_subvectors):
                bit_offset = j * nbits
                byte_offset = bit_offset // 8
                bit_shift = bit_offset % 8
                mask = (1 << nbits) - 1
                codes[:, j] = (packed[:, byte_offset].astype(np.int64) >> bit_shift) & mask
                # Handle cross-byte boundary
                if bit_shift + nbits > 8:
                    remaining = bit_shift + nbits - 8
                    codes[:, j] |= (packed[:, byte_offset + 1].astype(np.int64) & ((1 << remaining) - 1)) << (nbits - remaining)
            return codes
        else:
            # nbits > 8, use 2 bytes per code
            codes = np.zeros((n, self.n_subvectors), dtype=np.int64)
            bytes_per_code = (nbits + 7) // 8
            for j in range(self.n_subvectors):
                start = j * bytes_per_code
                for b in range(bytes_per_code):
                    codes[:, j] |= packed[:, start + b].astype(np.int64) << (8 * b)
                codes[:, j] &= (1 << nbits) - 1
            return codes

    def _pack_codes(self, codes: np.ndarray) -> np.ndarray:
        """Pack (N, m) per-subvector indices back to FAISS format."""
        n = codes.shape[0]
        nbits = int(np.log2(self.n_clusters_per_sub))

        if nbits == 8:
            return codes.astype(np.uint8).reshape(n, self.n_subvectors)
        elif nbits < 8:
            total_bits = self.n_subvectors * nbits
            total_bytes = (total_bits + 7) // 8
            packed = np.zeros((n, total_bytes), dtype=np.uint8)
            for j in range(self.n_subvectors):
                bit_offset = j * nbits
                byte_offset = bit_offset // 8
                bit_shift = bit_offset % 8
                packed[:, byte_offset] |= (codes[:, j].astype(np.uint8) << bit_shift)
                if bit_shift + nbits > 8:
                    packed[:, byte_offset + 1] |= (codes[:, j].astype(np.uint8) >> (8 - bit_shift))
            return packed
        else:
            bytes_per_code = (nbits + 7) // 8
            packed = np.zeros((n, self.n_subvectors * bytes_per_code), dtype=np.uint8)
            for j in range(self.n_subvectors):
                start = j * bytes_per_code
                for b in range(bytes_per_code):
                    packed[:, start + b] = (codes[:, j] >> (8 * b)).astype(np.uint8)
            return packed

    def generate_semantic_ids(self, embeddings: torch.Tensor) -> List[str]:
        """Generate semantic ID strings from embeddings.

        Returns list of "c1_c2_..._cm" strings.
        """
        codes = self.encode(embeddings)  # (N, m)
        n_samples = codes.shape[0]
        semantic_ids = []
        for i in range(n_samples):
            sid = "_".join(str(codes[i, j]) for j in range(self.n_subvectors))
            semantic_ids.append(sid)
        return semantic_ids

    def get_subspace_centroids(self) -> List[np.ndarray]:
        """Get per-subspace centroids. Returns list of m arrays, each (M, sub_dim)."""
        return [self._codebooks[j] for j in range(self.n_subvectors)]

    def get_fullspace_centroids(self) -> List[torch.Tensor]:
        """Get per-subspace centroids projected back to original D-dim space.

        Each subspace's centroids are zero-padded in rotated space then
        inverse-rotated to original space. Sum across all subspaces gives
        the full reconstruction.

        Returns: list of m tensors, each (M, D)
        """
        full_centroids = []
        for j in range(self.n_subvectors):
            # Zero-pad in rotated space
            padded = np.zeros(
                (self.n_clusters_per_sub, self.n_features), dtype=np.float32
            )
            start = j * self.sub_dim
            end = start + self.sub_dim
            padded[:, start:end] = self._codebooks[j]
            # Inverse rotate to original space
            original = padded @ self._rotation  # R_inv = R.T, but FAISS stores it transposed
            full_centroids.append(torch.tensor(original, dtype=torch.float32))
        return full_centroids

    def save(self, path: str):
        """Save model to file."""
        model_data = {
            'model_type': 'opq',
            'n_features': self.n_features,
            'n_subvectors': self.n_subvectors,
            'n_clusters_per_sub': self.n_clusters_per_sub,
            'normalize_input': self.normalize_input,
            'rotation': self._rotation,
            'codebooks': self._codebooks,
        }
        torch.save(model_data, path)
        print(f"OPQ model saved to {path}")

    @classmethod
    def from_saved(cls, model_data: dict) -> 'OPQQuantizer':
        """Load from saved model_data dict."""
        import faiss

        q = cls(
            n_features=model_data['n_features'],
            n_subvectors=model_data['n_subvectors'],
            n_clusters_per_sub=model_data['n_clusters_per_sub'],
            normalize_input=model_data.get('normalize_input', True),
        )
        q._rotation = model_data['rotation']
        q._codebooks = model_data['codebooks']

        # Rebuild FAISS objects
        nbits = int(np.log2(q.n_clusters_per_sub))
        opq = faiss.OPQMatrix(q.n_features, q.n_subvectors)
        faiss.copy_array_to_vector(q._rotation.ravel().astype(np.float32), opq.A)
        opq.is_trained = True

        pq = faiss.ProductQuantizer(q.n_features, q.n_subvectors, nbits)
        faiss.copy_array_to_vector(q._codebooks.ravel().astype(np.float32), pq.centroids)
        pq.is_trained = True

        opq.pq = pq
        q.opq_matrix = opq
        q.pq = pq

        return q
