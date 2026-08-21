"""Phase-1 ordinal regression heads.

All heads share the same 2-layer projector architecture to ensure
capacity-matched comparison. Only the final readout/geometric constraint differs.

Projector: Linear(in_dim, proj_dim) → ReLU → Dropout → Linear(proj_dim, proj_dim)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_projector(in_dim, proj_dim, dropout=0.3):
    return nn.Sequential(
        nn.Linear(in_dim, proj_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(proj_dim, proj_dim),
    )


class EuclideanHuberHead(nn.Module):
    """E01A — matched-capacity scalar regression.

    Same projector as FOROH but no L2 normalization and no angular readout.
    The scalar is intentionally NOT clamped during training. Clipping/rounding
    happens only during evaluation so out-of-range predictions keep gradients.

    The final readout is bias-free, giving exactly ``proj_dim`` direction
    parameters — the same count as FOROH's learnable axis ``w``.
    """

    METHOD = "euclidean_huber"

    def __init__(self, in_dim, proj_dim=128, c_max=3, dropout=0.3):
        super().__init__()
        self.c_max = c_max
        self.projector = _make_projector(in_dim, proj_dim, dropout)
        self.readout = nn.Linear(proj_dim, 1, bias=False)

    def forward(self, z):
        h = self.projector(z)
        score = self.readout(h).squeeze(-1)
        return {"score": score}


class NormalizedCosineHead(nn.Module):
    """E01B — cosine regression on the hypersphere.

    Same projector → L2-normalize → score = ((1 − u·w) / 2) × C_max.
    Identical trainable capacity to FOROH; only the score map differs.
    """

    METHOD = "normalized_cosine"

    def __init__(self, in_dim, proj_dim=128, c_max=3, dropout=0.3):
        super().__init__()
        self.c_max = c_max
        self.projector = _make_projector(in_dim, proj_dim, dropout)
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))

    def _get_w(self):
        return F.normalize(self.w_raw, dim=0)

    def forward(self, z):
        u = F.normalize(self.projector(z), dim=-1)
        w = self._get_w()
        with torch.amp.autocast("cuda", enabled=False):
            u32, w32 = u.float(), w.float()
            cos = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            score = ((1.0 - cos) / 2.0) * self.c_max
        return {"score": score, "u": u, "cos": cos}


class FOROHHead(nn.Module):
    """E01C / E02B — FOROH angular regression (level-set).

    Same projector → L2-normalize → θ = arccos(u·w) → score = (θ/π) × C_max.
    Only the polar coordinate is supervised; azimuth remains free.
    """

    METHOD = "foroh"

    def __init__(self, in_dim, proj_dim=128, c_max=3, dropout=0.3):
        super().__init__()
        self.c_max = c_max
        self.projector = _make_projector(in_dim, proj_dim, dropout)
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))

    def _get_w(self):
        return F.normalize(self.w_raw, dim=0)

    def forward(self, z):
        u = F.normalize(self.projector(z), dim=-1)
        w = self._get_w()
        with torch.amp.autocast("cuda", enabled=False):
            u32, w32 = u.float(), w.float()
            cos = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            score = (theta / math.pi) * self.c_max
        return {"score": score, "u": u, "theta": theta}


class PointPrototypeHead(nn.Module):
    """E02A — hyperspherical point-prototype control.

    All grade prototypes lie on one meridian:
      a_y = cos(θ_y) w + sin(θ_y) q,   q ⟂ w,  θ_y = πy/C_max

    FOROH and this control have the same trainable projector and the same
    trainable severity axis ``w``. ``q`` is derived from a fixed random buffer,
    so the point control does not gain an extra learnable direction.

    The training engine constrains the full geodesic distance to ``a_y``;
    inference still reads the ordinal score from polar angle to ``w``.
    """

    METHOD = "point_prototype"

    def __init__(self, in_dim, proj_dim=128, c_max=3, dropout=0.3):
        super().__init__()
        self.c_max = c_max
        self.projector = _make_projector(in_dim, proj_dim, dropout)
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))
        self.register_buffer("q_ref", F.normalize(torch.randn(proj_dim), dim=0))

    def _get_w(self):
        return F.normalize(self.w_raw, dim=0)

    def _get_q(self, w):
        q = self.q_ref - (self.q_ref @ w) * w
        if q.norm() < 1e-6:
            basis = torch.zeros_like(w)
            basis[torch.argmin(w.abs())] = 1.0
            q = basis - (basis @ w) * w
        return F.normalize(q, dim=0)

    def get_prototypes(self):
        w = self._get_w()
        q = self._get_q(w)
        thetas = torch.arange(
            self.c_max + 1, device=w.device, dtype=w.dtype
        ) * (math.pi / self.c_max)
        return (
            thetas.cos().unsqueeze(1) * w.unsqueeze(0)
            + thetas.sin().unsqueeze(1) * q.unsqueeze(0)
        )

    def forward(self, z):
        u = F.normalize(self.projector(z), dim=-1)
        w = self._get_w()
        with torch.amp.autocast("cuda", enabled=False):
            u32, w32 = u.float(), w.float()
            cos_w = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos_w)
            score = (theta / math.pi) * self.c_max
        prototypes = self.get_prototypes()
        return {"score": score, "u": u, "theta": theta, "prototypes": prototypes}


HEAD_REGISTRY = {
    "euclidean_huber": EuclideanHuberHead,
    "normalized_cosine": NormalizedCosineHead,
    "foroh": FOROHHead,
    "point_prototype": PointPrototypeHead,
}
