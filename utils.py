import torch
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

MODALITIES = [0, 1, 2, 3]
SUBSETS = [list(s) for r in range(1, 5) for s in combinations(MODALITIES, r)]  # all non-empty subsets

#Credit: https://github.com/ReubenDo/U-HVED/blob/master/extensions/u_hved/application.py
#UHVED Paper code
def KL_divergence(mu_1, logvar_1, mu_2, logvar_2):
    """KLD(p_1 || p_2)"""
    var_1 = torch.exp(logvar_1)
    var_2 = torch.exp(logvar_2)
    KLD = 0.5 * torch.mean(-1 + logvar_2 - logvar_1 + (var_1 + (mu_1 - mu_2)**2) / (var_2 + 1e-7))
    
    return KLD


def Product_Gaussian(means, logvars, subset):
    eps = 1e-7
    mu_prior = torch.zeros_like(means[0])
    log_prior = torch.zeros_like(means[0])

    T = [1 / (torch.exp(logvars[i]) + eps) for i in subset] + [1 + log_prior]
    mu = [means[i] / (torch.exp(logvars[i]) + eps) for i in subset] + [mu_prior]

    T_sum = torch.stack(T).sum(dim=0)
    mu_sum = torch.stack(mu).sum(dim=0)

    posterior_means = mu_sum / T_sum
    var = 1 / T_sum
    posterior_logvars = torch.log(var + eps)
    posterior_logvars = torch.clamp(posterior_logvars, min=-10, max=10)

    return posterior_means, posterior_logvars


def compute_KLD(means, logvars):
    """
    means:   list of 4 tensors, one per modality
    logvars: list of 4 tensors, one per modality
    """
    assert len(means) == len(logvars) == 4

    mu_prior = torch.zeros_like(means[0])
    log_prior = torch.zeros_like(means[0])

    full_means, full_logvars = Product_Gaussian(means, logvars, MODALITIES)
    full_logvars = torch.clamp(full_logvars, min=-10, max=10)

    sum_inter_KLD = 0
    sum_prior_KLD = 0

    for subset in SUBSETS:
        sub_means, sub_logvars = Product_Gaussian(means, logvars, subset)

        sum_inter_KLD += KL_divergence(full_means, full_logvars, sub_means, sub_logvars)
        sum_prior_KLD += KL_divergence(sub_means, sub_logvars, mu_prior, log_prior)

    prior = torch.clamp(sum_prior_KLD, min=0.5)

    return sum_inter_KLD / 14, sum_prior_KLD / 15


def kl_weight(epoch, warmup_epochs=100, max_weight=0.01):
    return max_weight * min(1.0, epoch / warmup_epochs)

def show_tensor(tensor, title=None, cmap='gray'):
    img = tensor.detach().cpu()
    
    # handle (C, H, W) -> (H, W, C) or squeeze channel dim if C=1
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img.squeeze(0)
        else:
            img = img.permute(1, 2, 0)
    
    # normalize to [0, 1] for display
    img = img.float()
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap if img.ndim == 2 else None)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()


def visualize_mosaic(I, P, L, alpha=0.4, save_path=None):
    """
    I: (4, H, W, D) - image tensor, first channel used
    P: (3, H, W, D) - prediction binary masks
    L: (3, H, W, D) - ground truth binary masks
    """
    d_slice = I.shape[-1] // 2

    img  = I[0, :, :, d_slice].cpu().float().numpy()
    pred = P[:, :, :, d_slice].cpu().float().numpy()
    gt   = L[:, :, :, d_slice].cpu().float().numpy()

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    colors = [
        [1.0, 0.2, 0.2],   # TC  — red
        [0.2, 1.0, 0.2],   # WT  — green
        [0.2, 0.6, 1.0],   # ET  — blue
    ]
    class_names = ['Tumor Core (TC)', 'Whole Tumor (WT)', 'Enhancing Tumor (ET)']

    def make_overlay(base_img, masks, class_idx=None):
        rgb = np.stack([base_img] * 3, axis=-1)
        indices = [class_idx] if class_idx is not None else range(len(colors))
        for c in indices:
            mask = masks[c].astype(bool)
            for ch in range(3):
                rgb[:, :, ch] = np.where(
                    mask,
                    (1 - alpha) * rgb[:, :, ch] + alpha * colors[c][ch],
                    rgb[:, :, ch]
                )
        return np.clip(rgb, 0, 1)

    def dice(p, g):
        intersection = (p * g).sum()
        return (2 * intersection / (p.sum() + g.sum() + 1e-8)).item()

    dice_scores = [dice(pred[c], gt[c]) for c in range(3)]

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.patch.set_facecolor('#1a1a1a')

    # ── Row 0: summary overlays ───────────────────────────────────────────
    row0 = [
        ('Image',        np.stack([img] * 3, axis=-1)),
        ('Ground truth', make_overlay(img, gt)),
        ('Prediction',   make_overlay(img, pred)),
    ]
    for ax, (title, overlay) in zip(axes[0], row0):
        ax.imshow(overlay, interpolation='nearest')
        ax.set_title(title, color='white', fontsize=18, pad=8)
        ax.axis('off')
        ax.set_facecolor('#1a1a1a')

    # ── Rows 1–2: per-class overlays ──────────────────────────────────────
    row_configs = [
        (axes[1], gt,   'GT'),
        (axes[2], pred, 'Pred'),
    ]
    for axes_row, masks, label in row_configs:
        for c, ax in enumerate(axes_row):
            overlay = make_overlay(img, masks, class_idx=c)
            ax.imshow(overlay, interpolation='nearest')
            ax.set_title(
                f'{label} — {class_names[c]}\ndice: {dice_scores[c]:.3f}',
                color='white', fontsize=18, pad=8
            )
            ax.axis('off')
            ax.set_facecolor('#1a1a1a')

    # ── Row labels ────────────────────────────────────────────────────────
    for ax, label in zip(axes[:, 0], ['Overview', 'GT per class', 'Pred per class']):
        ax.set_ylabel(label, color='#aaaaaa', fontsize=18, labelpad=8)

    # ── Legend ────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=colors[c], label=f'{class_names[c]}  (dice: {dice_scores[c]:.3f})')
        for c in range(3)
    ]
    fig.legend(
        handles=patches,
        loc='lower center',
        ncol=3,
        fontsize=20,
        facecolor='#2a2a2a',
        edgecolor='#444',
        labelcolor='white',
        framealpha=1,
        bbox_to_anchor=(0.5, -0.03)
    )

    plt.suptitle(f'BraTS segmentation  |  slice d={d_slice}', color='white', fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), dpi=150)

    plt.show()
    #return fig