import torch
from torch import nn


class AffinityGHead(nn.Module):
    """Trainable head on top of a pre-computed pooled representation `g`
    and/or an assay-context embedding `c`.

    Supported configurations:
      - f(g)              : use_g=True,  assay_context_dim=0    (g-only baseline)
      - f(g || c)         : use_g=True,  assay_context_dim>0, fusion="concat"
      - f(c)              : use_g=False, assay_context_dim>0, fusion="concat"
      - f(FiLM(g; c))     : use_g=True,  assay_context_dim>0, fusion="film"

    `fusion` controls how `c` is integrated with `g`:
      - "concat": x = [g; c]                                  (default, original)
      - "film":   gamma, beta = W_g(c), W_b(c)
                  x = gamma * g + beta
                  Requires use_g=True AND assay_context_dim>0.
                  Initialized so FiLM = identity at step 0
                  (zero weights; gamma bias=1, beta bias=0).

    Input:
        g:               FloatTensor[B, token_z]              (required if use_g)
        assay_context:   FloatTensor[B, assay_context_dim]    (required if assay_context_dim>0)
        no_context_mask: BoolTensor[B]                        (optional)
            When provided, rows where the mask is True get their context
            slot overwritten by a learned null vector — used for assays
            (e.g. ITC) that we do not want to condition on a real context.
            Applied before the concat/FiLM fusion.

    Output dict:
        affinity_pred_value:    FloatTensor[B, 1]
        affinity_logits_binary: FloatTensor[B, 1]
    """

    def __init__(
        self,
        token_z: int = 128,
        input_token_s: int = 384,
        assay_context_dim: int = 0,
        use_g: bool = True,
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        if fusion not in {"concat", "film"}:
            raise ValueError(f"fusion must be 'concat' or 'film', got {fusion!r}")
        self.use_g = use_g
        self.use_assay_context = assay_context_dim > 0
        self.assay_context_dim = assay_context_dim
        self.fusion = fusion

        if not self.use_g and not self.use_assay_context:
            raise ValueError(
                "AffinityGHead needs at least one of use_g=True or assay_context_dim>0"
            )

        if fusion == "film" and not (self.use_g and self.use_assay_context):
            raise ValueError(
                "fusion='film' requires both use_g=True and assay_context_dim>0"
            )

        if fusion == "film":
            fused_dim = token_z
        else:
            fused_dim = (token_z if self.use_g else 0) + (
                assay_context_dim if self.use_assay_context else 0
            )

        if self.use_assay_context:
            self.null_context = nn.Parameter(torch.zeros(assay_context_dim))

        if fusion == "film":
            self.to_gamma = nn.Linear(assay_context_dim, token_z)
            self.to_beta = nn.Linear(assay_context_dim, token_z)
            nn.init.zeros_(self.to_gamma.weight)
            nn.init.ones_(self.to_gamma.bias)
            nn.init.zeros_(self.to_beta.weight)
            nn.init.zeros_(self.to_beta.bias)

        self.affinity_out_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, input_token_s),
            nn.ReLU(),
        )

        self.to_affinity_pred_value = nn.Sequential(
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        )

        self.to_affinity_pred_score = nn.Sequential(
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        )
        self.to_affinity_logits_binary = nn.Linear(1, 1)

    def forward(
        self,
        g: torch.Tensor | None = None,
        assay_context: torch.Tensor | None = None,
        no_context_mask: torch.Tensor | None = None,
    ):
        if self.use_g and g is None:
            raise ValueError("g is required when use_g=True")
        if self.use_assay_context and assay_context is None:
            raise ValueError("assay_context is required when assay_context_dim > 0")

        if self.use_assay_context and no_context_mask is not None:
            null = self.null_context.to(assay_context.dtype).unsqueeze(0).expand_as(assay_context)
            assay_context = torch.where(
                no_context_mask.unsqueeze(-1), null, assay_context
            )

        if self.fusion == "film":
            gamma = self.to_gamma(assay_context)
            beta = self.to_beta(assay_context)
            x = gamma * g + beta
        else:
            parts = []
            if self.use_g:
                parts.append(g)
            if self.use_assay_context:
                parts.append(assay_context)
            x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

        h = self.affinity_out_mlp(x)
        affinity_pred_value = self.to_affinity_pred_value(h).reshape(-1, 1)
        affinity_pred_score = self.to_affinity_pred_score(h).reshape(-1, 1)
        affinity_logits_binary = self.to_affinity_logits_binary(
            affinity_pred_score
        ).reshape(-1, 1)
        return {
            "affinity_pred_value": affinity_pred_value,
            "affinity_logits_binary": affinity_logits_binary,
        }
