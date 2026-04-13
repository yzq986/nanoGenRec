"""
Reconstruction Loss Metric

Measures quantization precision by computing L2 distance between
original embeddings and their reconstructed versions.

Reference: OneRec (arxiv 2506.13695)
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F

from .base import BaseMetric, MetricResult


class ReconstructionLossMetric(BaseMetric):
    """Reconstruction Loss: L2(x - x_hat)

    Measures how well the RKMeans model can reconstruct the original embeddings.
    Lower values indicate better quantization precision.

    Computation (matches rkmeans_stage2_train_v2.py generate_semantic_ids):
    1. normalize_residuals=True: L2 normalize input embeddings once (layer 0 only)
    2. For each layer: assign to nearest centroid, subtract centroid to get residual
    3. Residuals are NOT re-normalized between layers (raw scale preserved)
    4. Final reconstruction = sum of all layer centroids
    5. Loss = mean(||x_normalized - x_hat||^2)
    """

    name = 'reconstruction_loss'
    requires_model = True
    requires_semantic_ids = False

    # Lower is better
    thresholds = {
        'excellent': 0.05,
        'good': 0.10,
        'acceptable': 0.20,
    }

    def compute(
        self,
        embeddings: torch.Tensor,
        model: Optional[Any] = None,
        semantic_ids: Optional[List[str]] = None,
        layer_assignments: Optional[List[torch.Tensor]] = None,
        normalize_residuals: bool = True,
        chunk_size: int = 50000,
        **kwargs
    ) -> MetricResult:
        """Compute reconstruction loss

        Args:
            embeddings: (N, D) tensor of original embeddings
            model: RKMeans model with kmeans_layers attribute
            layer_assignments: Optional pre-computed layer assignments
            normalize_residuals: If True, L2 normalize input once (layer 0 only).
                                 Residuals are NOT re-normalized between layers.
            chunk_size: Batch size for processing

        Returns:
            MetricResult with total loss and per-layer losses
        """
        self.validate_inputs(embeddings, model, semantic_ids)

        n_samples = embeddings.shape[0]
        device = next(iter(model.kmeans_layers[0].centroids.device for _ in [0]))

        # Match generate_semantic_ids: normalize input once (layer 0), then raw residuals
        layer_losses = []
        current_residuals = embeddings.clone()

        # Normalize input embeddings once (layer 0 only)
        if normalize_residuals:
            normalized_input = []
            for i in range(0, n_samples, chunk_size):
                chunk = current_residuals[i:i+chunk_size].to(device)
                chunk = F.normalize(chunk, p=2, dim=1).cpu()
                normalized_input.append(chunk)
            current_residuals = torch.cat(normalized_input, dim=0)

        # This is the normalized input we compare against for total loss
        input_for_loss = current_residuals.clone()
        reconstructed = torch.zeros_like(current_residuals)

        for layer_idx, kmeans in enumerate(model.kmeans_layers):
            # No per-layer normalization — residuals keep raw scale

            # Get assignments
            if layer_assignments is not None:
                assignments = layer_assignments[layer_idx]
            else:
                all_assignments = []
                for i in range(0, n_samples, chunk_size):
                    chunk = current_residuals[i:i+chunk_size].to(device)
                    batch_assignments = kmeans.predict(chunk).cpu()
                    all_assignments.append(batch_assignments)
                assignments = torch.cat(all_assignments, dim=0)

            # Compute layer reconstruction (centroid directly, no scaling)
            layer_reconstruction = kmeans.centroids[assignments].cpu()

            reconstructed += layer_reconstruction

            # Residual for next layer = current - centroid (raw scale)
            current_residuals = current_residuals - layer_reconstruction
            layer_loss = torch.norm(current_residuals, dim=1).pow(2).mean().item()
            layer_losses.append(layer_loss)

        # Total reconstruction loss
        total_loss = torch.norm(input_for_loss - reconstructed, dim=1).pow(2).mean().item()

        # Normalized loss (relative to input norm after optional normalization)
        input_norm = torch.norm(input_for_loss, dim=1).pow(2).mean().item()
        normalized_loss = total_loss / (input_norm + 1e-8)

        status = self.assess_quality(normalized_loss)

        return MetricResult(
            name=self.name,
            value=normalized_loss,
            layer_values=layer_losses,
            details={
                'total_loss': total_loss,
                'normalized_loss': normalized_loss,
                'input_norm_mean': input_norm,
                'input_normalized': normalize_residuals,
                'n_layers': len(layer_losses),
                'n_samples': n_samples,
            },
            status=status,
        )
