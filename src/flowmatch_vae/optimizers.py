"""Muon optimizer (Newton-Schulz matrix orthogonalization) + SGD.

Based on:
  - "Muon is Scalable for LLM Training" (arXiv:2502.16982)
  - Keller Jordan's original Muon blog post

Muon is applied to 2D weight matrices; all other parameters (biases,
LayerNorm/RMSNorm, embeddings, output heads) use standard SGD.
"""

from __future__ import annotations

import torch
from torch.optim import SGD


# ---------------------------------------------------------------------------
# Newton-Schulz iteration
# ---------------------------------------------------------------------------

_NS_COEFFS = (3.4445, -4.7750, 2.0315)  # a, b, c — quintic polynomial coefficients


@torch.no_grad()
def _newton_schulz(M: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise a 2-D matrix via Newton-Schulz iteration.

    Solves ``(M M^T)^{-1/2} M ≈ U V^T`` (the orthogonal part of SVD).

    Args:
        M: Input 2-D tensor (gradient momentum).
        steps: Number of Newton-Schulz iterations (default 5).
        eps: Small constant for numerical stability in Frobenius norm.

    Returns:
        Approximately orthogonal matrix of the same shape as *M*.
    """
    a, b, c = _NS_COEFFS
    X = M.bfloat16()
    X = X / (X.norm() + eps)

    # Transpose to make the matrix wide (reduces matmul cost of X @ X^T)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    """Muon — momentum orthogonalised by Newton-Schulz.

    For **2-D weight matrices only** (linear / conv weight tensors).
    Non-matrix parameters should use :class:`torch.optim.SGD` instead.

    Key hyper-parameters (sensible defaults from the paper):
        lr=1e-4, momentum=0.95, weight_decay=0.1, ns_steps=5
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            ns_steps=ns_steps, eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                assert g.ndim == 2, (
                    f"Muon only supports 2-D parameters, got shape {g.shape}. "
                    "Use SGD for non-matrix params."
                )

                state = self.state[p]

                # Initialise momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                m = state["momentum_buffer"]

                # Standard momentum: m = β m + g
                m.mul_(mu).add_(g)

                # Nesterov lookahead input for orthogonalisation: β m + g
                g_ortho = mu * m + g

                # Newton-Schulz orthogonalisation
                O = _newton_schulz(g_ortho, steps=ns_steps, eps=eps)

                # Per-parameter scaling: 0.2 * √max(A, B)
                A_dim, B_dim = p.shape
                scale = 0.2 * (max(A_dim, B_dim) ** 0.5)

                # W -= lr * (scale * O + wd * W)
                p.mul_(1.0 - lr * wd)
                p.add_(O, alpha=-lr * scale)

        return loss


# ---------------------------------------------------------------------------
# Helper: split model parameters into Muon / SGD groups
# ---------------------------------------------------------------------------

# Names whose 2-D weight should NOT go to Muon (use SGD instead).
# These are substrings matched against parameter names.
_MUON_EXCLUDE_SUBSTRS = ("patch_embed", "out_proj", "mu_head", "logvar_head", "z_proj")


def split_param_groups(model: torch.nn.Module, lr: float = 1e-4,
                       momentum: float = 0.95, weight_decay: float = 0.1):
    """Split model parameters into Muon (2-D matrices) and SGD (everything else).

    Returns a list of two ``dict`` s suitable for passing to the constructors of
    :class:`Muon` and :class:`SGD` respectively.

    Args:
        model: The model whose parameters to split.
        lr: Learning rate (shared by both optimisers).
        momentum: Momentum for Muon. SGD uses a fixed 0.9.
        weight_decay: Weight decay for both.

    Returns:
        ``(muon_params, sgd_params)`` — each is a dict with ``params`` and
        optimiser-specific hyper-parameters.
    """
    muon_params = []
    sgd_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Exclude certain layers even if they are 2-D
        excluded = any(s in name for s in _MUON_EXCLUDE_SUBSTRS)

        if p.ndim == 2 and not excluded:
            muon_params.append(p)
        else:
            sgd_params.append(p)

    muon_group = dict(params=muon_params, lr=lr, momentum=momentum,
                      weight_decay=weight_decay)
    sgd_group = dict(params=sgd_params, lr=lr, momentum=0.9,
                     weight_decay=weight_decay)
    return muon_group, sgd_group
