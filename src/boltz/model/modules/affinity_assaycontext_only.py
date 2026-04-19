import torch
from torch import nn


class AffinityAssayContextOnlyModule(nn.Module):
    """Simple MLP that predicts affinity from assay context embedding alone."""

    def __init__(
        self,
        assay_context_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(assay_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, assay_context: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        assay_context : Tensor
            Assay context embedding of shape [B, assay_context_dim].

        Returns
        -------
        dict[str, Tensor]
            Dictionary with key "affinity_pred_value" of shape [B, 1].
        """
        pred = self.mlp(assay_context)
        return {"affinity_pred_value": pred}
