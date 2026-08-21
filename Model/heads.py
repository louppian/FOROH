"""Phase-1 ordinal regression heads.

All heads share the same 2-layer projector architecture to ensure
capacity-matched comparison. Only the final readout differs.

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
    """E01A — Matched-capacity scalar regression.

    Same projector as FOROH but NO normalization and NO angular readout.
    score = Linear(proj_dim, 1), clamped to [0, C_max].
    """

    METHOD = "euclidean_huber"

    def __init__(self, in_dim, proj_dim=128, c_max=3, dropout=0.3):
        super().__init__()
        self.c_max = c_max
        self.projector = _make_projector(in_dim, proj_dim, dropout)
        self.readout = nn.Linear(proj_dim, 1)

    def forward(self, z):
        h = self.projector(z)
        score = self.readout(h).squeeze(-1).clamp(0, self.c_max)
        return {"score": score}


class NormalizedCosineHead(nn.Module):
    """E01B — Cosine regression on the hypersphere.

    Same projector → L2-normalize → score = ((1 − u·w) / 2) × C_max.
    Identical to FOROH except the monotonic map from cosine to score is
    linear in cos rather than linear in angle.
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
    Only θ is supervised; azimuth v is free (level-set representation).
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
    """E02A — Hyperspherical point-prototype ordinal regression.

    All grade prototypes lie on the same meridian:
      a_y = cos(θ_y) w + sin(θ_y) q,   q ⊥ w,  θ_y = πy/C_max

    Both polar angle AND azimuth are supervised (contrast with FOROH
    where only polar angle is supervised and azimuth is free).

    Loss = Huber(score, y) + proto_lambda × mean(1 − u · a_y).
    """

    METHOD = "point_prototype"

    def __init__(self, in_dim, proj_dim=128, c_max=3, dropout=0.3):
        super().__init__()
        self.c_max = c_max
        self.projector = _make_projector(in_dim, proj_dim, dropout)
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))
        q_init = torch.randn(proj_dim)
        q_init = q_init - (q_init @ self.w_raw.data) * self.w_raw.data
        self.q_raw = nn.Parameter(F.normalize(q_init, dim=0))

    def _get_w(self):
        return F.normalize(self.w_raw, dim=0)

    def _get_q(self, w):
        q = self.q_raw - (self.q_raw @ w) * w
        return F.normalize(q, dim=0)

    def get_prototypes(self):
        w = self._get_w()
        q = self._get_q(w)
        thetas = torch.tensor(
            [math.pi * y / self.c_max for y in range(self.c_max + 1)],
            device=w.device, dtype=w.dtype,
        )
        return (thetas.cos().unsqueeze(1) * w.unsqueeze(0)
                + thetas.sin().unsqueeze(1) * q.unsqueeze(0))

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
