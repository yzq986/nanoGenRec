"""Finite Scalar Quantization (FSQ) layer with PCA projection.

Non-learned FSQ: projects residuals to low-dim via PCA, then quantizes each
dimension to a fixed set of discrete levels. Codebook = Cartesian product of
per-dimension levels (no codebook collapse by construction).

Reference: Mentzer et al., "Finite Scalar Quantization" (arxiv 2309.15505)
"""

from typing import List, Optional

import torch

# Pre-defined level configs from FSQ paper Table 3.
# Keys are "{d}d_{codebook_size}" for easy CLI reference.
FSQ_LEVEL_CONFIGS = {
    '4d_4096':  [8, 8, 8, 8],        # 4096 exact
    '5d_4375':  [7, 5, 5, 5, 5],     # 4375 ~ 2^12
    '6d_4096':  [4, 4, 4, 4, 4, 4],  # 4096 exact
    '5d_6000':  [8, 6, 5, 5, 5],     # 6000
}


def _codebook_size(levels: List[int]) -> int:
    s = 1
    for l in levels:
        s *= l
    return s


class FSQLayer:
    """Non-learned FSQ quantizer with PCA dimensionality reduction.

    Pipeline:
        D-dim residuals -> PCA project to d-dim -> per-dim tanh+round quantize
        -> mixed-radix encode to single integer code
    Inverse:
        code -> decode to d-dim quantized -> PCA inverse -> D-dim reconstruction
    """

    def __init__(self, levels: List[int], n_features: int):
        self.levels = levels
        self.d = len(levels)                    # low-dim target
        self.n_features = n_features            # original D
        self.codebook_size = _codebook_size(levels)

        # Mixed-radix basis for encoding: [1, L0, L0*L1, ...]
        self.basis = []
        b = 1
        for l in levels:
            self.basis.append(b)
            b *= l
        self.basis = torch.tensor(self.basis, dtype=torch.long)

        # PCA components (fitted during train)
        self.pca_components = None   # (d, D)
        self.pca_mean = None         # (D,)

    # ------------------------------------------------------------------
    # Training (fit PCA)
    # ------------------------------------------------------------------

    def train(self, residuals: torch.Tensor):
        """Fit PCA projection from residuals (N, D) -> (N, d).

        Uses torch.pca_lowrank for GPU-friendly randomized SVD.
        """
        N, D = residuals.shape
        assert D == self.n_features, f"Expected {self.n_features} features, got {D}"

        # Center
        self.pca_mean = residuals.mean(dim=0)
        centered = residuals - self.pca_mean

        # Randomized SVD -> top-d components
        U, S, V = torch.pca_lowrank(centered, q=self.d, niter=5)
        # V: (D, d) — columns are principal components
        self.pca_components = V.T  # (d, D)

        # Print stats
        projected = centered @ V   # (N, d)
        recon = projected @ V.T
        recon_err = (centered - recon).pow(2).sum(dim=1).mean().item()
        total_var = centered.pow(2).sum(dim=1).mean().item()
        explained = 1.0 - recon_err / (total_var + 1e-8)

        print(f"  FSQ PCA: {D}D -> {self.d}D, explained variance: {explained:.4f}")
        print(f"  FSQ codebook: {self.levels} = {self.codebook_size} codes")

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def _project(self, data: torch.Tensor) -> torch.Tensor:
        """Project D-dim data to d-dim PCA space."""
        centered = data - self.pca_mean.to(data.device)
        return centered @ self.pca_components.to(data.device).T  # (N, d)

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize d-dim projected vectors per FSQ paper.

        For each dimension with L levels:
          - Odd L:  values in {-L//2, ..., 0, ..., L//2}, offset = L//2
          - Even L: values in {-L//2+1, ..., L//2} via floor(L/2 * tanh(z)) + 0.5 shift
                    then round, giving {-L//2+1, ..., L//2}, offset = L//2 - 1

        Returns integer codes per dimension in [0, L-1].
        """
        N = z.shape[0]
        codes = torch.zeros(N, self.d, dtype=torch.long, device=z.device)

        for i, L in enumerate(self.levels):
            half = L // 2
            # tanh squash
            zi = half * torch.tanh(z[:, i])
            if L % 2 == 1:
                # Odd: round to {-half, ..., half}
                qi = torch.round(zi).long()
                codes[:, i] = qi + half  # shift to [0, L-1]
            else:
                # Even: shift by 0.5 to break symmetry, then round
                qi = torch.round(zi - 0.5).long() + 1  # {-half+1, ..., half}
                codes[:, i] = qi + half - 1  # shift to [0, L-1]

        return codes

    def _encode(self, per_dim_codes: torch.Tensor) -> torch.Tensor:
        """Mixed-radix encode d-dim codes to single integer."""
        basis = self.basis.to(per_dim_codes.device)
        return (per_dim_codes * basis).sum(dim=1)

    def _decode_index(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode single integer to d-dim per-dimension codes."""
        N = indices.shape[0]
        per_dim = torch.zeros(N, self.d, dtype=torch.long, device=indices.device)
        remaining = indices.clone()
        for i in range(self.d - 1, -1, -1):
            per_dim[:, i] = remaining // self.basis[i].item()
            remaining = remaining % self.basis[i].item()
        return per_dim

    def _per_dim_to_float(self, per_dim_codes: torch.Tensor) -> torch.Tensor:
        """Convert integer per-dim codes [0, L-1] back to quantized float values."""
        z_q = torch.zeros(per_dim_codes.shape[0], self.d,
                          dtype=torch.float32, device=per_dim_codes.device)
        for i, L in enumerate(self.levels):
            half = L // 2
            if L % 2 == 1:
                z_q[:, i] = per_dim_codes[:, i].float() - half
            else:
                z_q[:, i] = per_dim_codes[:, i].float() - half + 1
        return z_q

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, residuals: torch.Tensor) -> torch.Tensor:
        """Quantize residuals and return single integer code indices.

        Args:
            residuals: (N, D) tensor
        Returns:
            (N,) tensor of integer codes in [0, codebook_size)
        """
        z = self._project(residuals)         # (N, d)
        per_dim = self._quantize(z)          # (N, d) ints in [0, L_i)
        return self._encode(per_dim)         # (N,)

    def get_centroids_for_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Inverse map: code indices -> D-dim reconstructed vectors.

        Args:
            codes: (N,) integer codes
        Returns:
            (N, D) reconstructed vectors in original space
        """
        per_dim = self._decode_index(codes)          # (N, d)
        z_q = self._per_dim_to_float(per_dim)        # (N, d) quantized floats
        # Inverse PCA: d-dim -> D-dim
        recon = z_q @ self.pca_components.to(codes.device).float()  # (N, D)
        return recon + self.pca_mean.to(codes.device)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        return {
            'levels': self.levels,
            'n_features': self.n_features,
            'pca_components': self.pca_components.cpu(),
            'pca_mean': self.pca_mean.cpu(),
        }

    @classmethod
    def from_state(cls, state: dict) -> 'FSQLayer':
        layer = cls(state['levels'], state['n_features'])
        layer.pca_components = state['pca_components']
        layer.pca_mean = state['pca_mean']
        return layer
