"""
FOROH Visualization — t-SNE, θ scatter, confidence distribution
================================================================
Usage (서버에서):
  cd ~/JeongGeon/LOR
  python 5_visualize.py

Reads:
  outputs/exp1/lor_limuc/fold0.pt
  outputs/exp1/ce_limuc/fold0.pt

Saves to:
  outputs/figures/tsne_foroh_vs_ce.png
  outputs/figures/theta_vs_grade.png
  outputs/figures/confidence_distribution.png
"""

import sys, math, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── import from train code ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

# Dynamic import — handles both 3_train.py and train.py
def _import_train():
    p = Path(__file__).resolve().parent
    for name in ["3_train", "train"]:
        f = p / f"{name}.py"
        if f.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("trainmod", str(f))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("Cannot find 3_train.py or train.py")

T = _import_train()


# ════════════════════════════════════════
# Embedding extraction
# ════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, loader, method, head, c_max, device):
    """Extract embeddings, predictions, and labels from test set."""
    model.eval()
    all_z = []       # backbone features
    all_u = []       # projected embeddings (LOR only)
    all_scores = []  # continuous scores (LOR) or argmax (CE)
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        z = model.backbone(images)

        if method == "lor":
            score, u = model.head(z)
            all_u.append(u.cpu().float())
            all_scores.append(score.cpu().float())
        elif method == "ce":
            logits = model.head(z)
            all_scores.append(logits.argmax(-1).cpu().float())

        all_z.append(z.cpu().float())
        all_labels.append(labels)

    result = {
        "z": torch.cat(all_z).numpy(),
        "scores": torch.cat(all_scores).numpy(),
        "labels": torch.cat(all_labels).numpy(),
    }
    if all_u:
        result["u"] = torch.cat(all_u).numpy()
    return result


def load_model_from_ckpt(ckpt_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_dict = ckpt["args"]

    # Build args namespace
    args = argparse.Namespace(**args_dict)
    if not hasattr(args, "fixed_w"):
        args.fixed_w = False
    if not hasattr(args, "no_projector"):
        args.no_projector = False
    if not hasattr(args, "no_arccos"):
        args.no_arccos = False

    model, head, c_max = T.build_model(args)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model, head, c_max, args


# ════════════════════════════════════════
# Plot helpers
# ════════════════════════════════════════

GRADE_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
GRADE_NAMES = ["G0 (Normal)", "G1 (Mild)", "G2 (Moderate)", "G3 (Severe)"]


def plot_tsne_side_by_side(emb_foroh, labels_foroh, emb_ce, labels_ce, save_path):
    """t-SNE: FOROH (u ∈ S^{d-1}) vs CE (z ∈ R^d) side by side."""
    print("  Running t-SNE for FOROH...")
    tsne_foroh = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    xy_foroh = tsne_foroh.fit_transform(emb_foroh)

    print("  Running t-SNE for CE...")
    tsne_ce = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    xy_ce = tsne_ce.fit_transform(emb_ce)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, xy, labels, title in [
        (axes[0], xy_foroh, labels_foroh, "FOROH (projected u ∈ S$^{d-1}$)"),
        (axes[1], xy_ce, labels_ce, "CE (backbone z ∈ R$^d$)"),
    ]:
        for g in range(4):
            mask = labels == g
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=GRADE_COLORS[g], label=GRADE_NAMES[g],
                       s=8, alpha=0.5, edgecolors="none")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, markerscale=2.5)

    fig.suptitle("t-SNE: Embedding Space Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_theta_vs_grade(scores, labels, w_vec, u_emb, save_path):
    """θ vs GT grade scatter + boxplot."""
    # θ = arccos(u · w)
    cos_theta = np.clip(u_emb @ w_vec, -1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(cos_theta)
    theta_deg = np.degrees(theta)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: scatter
    ax = axes[0]
    for g in range(4):
        mask = labels == g
        jitter = np.random.normal(0, 0.12, mask.sum())
        ax.scatter(g + jitter, theta_deg[mask],
                   c=GRADE_COLORS[g], s=10, alpha=0.4, edgecolors="none")
    ax.set_xlabel("Ground Truth Grade", fontsize=12)
    ax.set_ylabel("θ (degrees)", fontsize=12)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["G0", "G1", "G2", "G3"])
    ax.set_title("θ = arccos(u · w) vs. GT Grade", fontsize=13, fontweight="bold")

    # grade boundaries
    for boundary in [0.5, 1.5, 2.5]:
        theta_b = boundary / 3.0 * 180  # C_max=3
        ax.axhline(theta_b, color="gray", ls="--", lw=0.8, alpha=0.6)

    # Right: boxplot
    ax2 = axes[1]
    data_by_grade = [theta_deg[labels == g] for g in range(4)]
    bp = ax2.boxplot(data_by_grade, labels=["G0", "G1", "G2", "G3"],
                     patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], GRADE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.set_ylabel("θ (degrees)", fontsize=12)
    ax2.set_title("Distribution", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confidence_distribution(u_emb, w_vec, labels, save_path):
    """Confidence = |u · w| distribution per grade."""
    cos_theta = u_emb @ w_vec
    confidence = np.abs(cos_theta)  # high |u·w| = close to pole = confident

    fig, ax = plt.subplots(figsize=(8, 5))
    for g in range(4):
        mask = labels == g
        ax.hist(confidence[mask], bins=30, alpha=0.5,
                color=GRADE_COLORS[g], label=GRADE_NAMES[g], density=True)

    ax.set_xlabel("|u · w| (confidence)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Geometric Confidence Distribution by Grade", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne_foroh_with_theta(u_emb, labels, w_vec, save_path):
    """t-SNE colored by θ (continuous) and by grade (discrete) side by side."""
    cos_theta = np.clip(u_emb @ w_vec, -1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(cos_theta)
    theta_norm = theta / np.pi  # 0~1, maps to score 0~C_max

    print("  Running t-SNE for θ-colored plot...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    xy = tsne.fit_transform(u_emb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by grade (discrete)
    ax = axes[0]
    for g in range(4):
        mask = labels == g
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=GRADE_COLORS[g], label=GRADE_NAMES[g],
                   s=10, alpha=0.5, edgecolors="none")
    ax.set_title("Colored by GT Grade", fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, markerscale=2.5)

    # Right: colored by θ (continuous)
    ax2 = axes[1]
    sc = ax2.scatter(xy[:, 0], xy[:, 1],
                     c=theta_norm, cmap="coolwarm", s=10, alpha=0.6, edgecolors="none",
                     vmin=0, vmax=1)
    cbar = plt.colorbar(sc, ax=ax2, fraction=0.04, pad=0.02)
    cbar.set_label("θ/π (→ score/C_max)", fontsize=10)
    ax2.set_title("Colored by θ (continuous score)", fontsize=13, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.suptitle("FOROH t-SNE: Grade vs. Continuous θ", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════
# Main
# ════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foroh-ckpt", type=str, default="outputs/exp1/lor_limuc/fold0.pt")
    parser.add_argument("--ce-ckpt", type=str, default="outputs/exp1/ce_limuc/fold0.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    parser.add_argument("--dataset", type=str, default="limuc")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load FOROH ──
    print("Loading FOROH model...")
    foroh_path = Path(args.foroh_ckpt)
    if not foroh_path.exists():
        print(f"  ERROR: {foroh_path} not found. Run exp1 first.")
        return
    model_f, head_f, c_max_f, args_f = load_model_from_ckpt(foroh_path, device)

    # ── Load CE ──
    print("Loading CE model...")
    ce_path = Path(args.ce_ckpt)
    if not ce_path.exists():
        print(f"  ERROR: {ce_path} not found. Run exp1 first.")
        return
    model_c, head_c, c_max_c, args_c = load_model_from_ckpt(ce_path, device)

    # ── Build test dataset ──
    print("Loading test dataset...")
    args_f.img_size = getattr(args_f, "img_size", 224)
    args_f.n_folds = getattr(args_f, "n_folds", 5)
    _, _, test_ds = T.build_datasets(args_f, fold=args.fold)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"  Test set: {len(test_ds)} images")

    # ── Extract embeddings ──
    print("\nExtracting FOROH embeddings...")
    emb_f = extract_embeddings(model_f, test_loader, "lor", head_f, c_max_f, device)

    print("Extracting CE embeddings...")
    emb_c = extract_embeddings(model_c, test_loader, "ce", head_c, c_max_c, device)

    # ── Get w vector ──
    w_vec = head_f._get_w().detach().cpu().numpy()

    # ── Class distribution ──
    counts = Counter(emb_f["labels"].tolist())
    print(f"\n  Class distribution: " + ", ".join(f"G{g}={counts.get(g,0)}" for g in range(4)))

    # ════════════════════════════════════════
    # Plot 1: t-SNE side by side (FOROH u vs CE z)
    # ════════════════════════════════════════
    print("\n[1/4] t-SNE: FOROH vs CE...")
    plot_tsne_side_by_side(
        emb_f["u"], emb_f["labels"],
        emb_c["z"], emb_c["labels"],
        out_dir / "tsne_foroh_vs_ce.png"
    )

    # ════════════════════════════════════════
    # Plot 2: θ vs GT grade
    # ════════════════════════════════════════
    print("\n[2/4] θ vs GT grade...")
    plot_theta_vs_grade(
        emb_f["scores"], emb_f["labels"], w_vec, emb_f["u"],
        out_dir / "theta_vs_grade.png"
    )

    # ════════════════════════════════════════
    # Plot 3: Confidence distribution
    # ════════════════════════════════════════
    print("\n[3/4] Confidence distribution...")
    plot_confidence_distribution(
        emb_f["u"], w_vec, emb_f["labels"],
        out_dir / "confidence_distribution.png"
    )

    # ════════════════════════════════════════
    # Plot 4: t-SNE with θ coloring
    # ════════════════════════════════════════
    print("\n[4/4] t-SNE with θ coloring...")
    plot_tsne_foroh_with_theta(
        emb_f["u"], emb_f["labels"], w_vec,
        out_dir / "tsne_foroh_theta.png"
    )

    print(f"\n✅ All figures saved to {out_dir}/")
    print(f"   tsne_foroh_vs_ce.png     — FOROH vs CE embedding comparison")
    print(f"   theta_vs_grade.png       — θ scatter + boxplot by grade")
    print(f"   confidence_distribution.png — |u·w| histogram per grade")
    print(f"   tsne_foroh_theta.png     — t-SNE colored by continuous θ")


if __name__ == "__main__":
    main()