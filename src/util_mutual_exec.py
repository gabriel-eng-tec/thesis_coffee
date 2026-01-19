"""
Mutual_Exc.py — Adaptación funcional para CDMENet (basado en el paper original)
Autor: Gabriel Barboza Álvarez (proyecto de conteo de frutos de café)
Compatibilidad: PyTorch 2.x

Este archivo implementa:
 - Pérdidas de consistencia y exclusión mutua entre ramas
 - Funciones auxiliares para entrenamiento semi-supervisado
 - Manejo de máscaras de densidad
 - Guardado de checkpoints
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 🔧 CONFIGURACIÓN GLOBAL — FÁCIL DE AJUSTAR
# ============================================================
DENSITY_THRESHOLDS = {
    "low": 0.0015,
    "mid": 0.0100
}

WEIGHTS = {
    "density_loss": 1.0,
    "class_loss": 0.01,
    "mutual_exc": 1.0,
    "semi_supervised": 0.01
}

EPS = 1e-6

# ============================================================
# 🧩 FUNCIONES AUXILIARES
# ============================================================

def save_checkpoint0(state, is_best, task_id):
    """Guarda el modelo actual y, si es el mejor, también como 'best'."""
    filename = f"./checkpoint_task_{task_id}.pth"
    torch.save(state, filename)
    if is_best:
        best_filename = f"./best_task_{task_id}.pth"
        torch.save(state, best_filename)
        print(f"[✔] Nuevo mejor modelo guardado en {best_filename}")


def densitymap_to_densitymask(density, threshold1=0.0, threshold2=0.01):
    """
    Convierte un mapa de densidad (float) en una máscara BINARIA (0 o 1) para una rama.
    Uso:
        mask2 = densitymap_to_densitymask(density, 0.0, low_thresh)   # rango "low"
        mask3 = densitymap_to_densitymask(density, low_thresh, mid_thresh) # rango "mid"
        mask4 = densitymap_to_densitymask(density, mid_thresh, +inf)   # rango "high"
    Devuelve: mask tipo long con valores {0,1}, shape = [N, H, W] (sin canal)
    """
    # density assumed shape [N,1,H,W] or [N,H,W] (so we unify)
    if density.dim() == 4 and density.size(1) == 1:
        d = density.squeeze(1)
    else:
        d = density

    device = d.device
    mask = torch.zeros_like(d, dtype=torch.long, device=device)
    # marcar 1 cuando density en el rango [threshold1, threshold2)
    if threshold2 is None:
        mask[d >= threshold1] = 1
    else:
        mask[(d >= threshold1) & (d < threshold2)] = 1
    return mask  # shape [N,H,W], dtype long (compatible con CrossEntropyLoss)


def dice_loss(pred, target, eps=1e-6):
    """Dice Loss para medir solapamiento entre máscaras (1 - Dice Coefficient)."""
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return 1 - num / den


# ============================================================
# 📘 PÉRDIDAS SUPERVISADAS
# ============================================================

def cross_entropy_loss(logits, target, ignore_index=10):
    """
    CrossEntropy estándar para máscaras discretas.
    """
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    return ce(logits, target)


# ============================================================
# 📘 EXCLUSIÓN MUTUA ENTRE RAMAS
# ============================================================

def mutual_exclusion_loss(logits2, logits3, logits4):
    """
    Penaliza correlación entre ramas usando similitud coseno sobre vectores por muestra.
    Procedimiento:
      - Para cada par, 'aplanamos' (C*H*W) por muestra, normalizamos y calculamos cosθ.
      - Tomamos la media de |cosθ| sobre el batch y promediamos las tres parejas.
    Retorna un escalar (torch.Tensor).
    """
    def cos_per_sample(a, b):
        # a, b shapes: [N, C, H, W]
        N = a.shape[0]
        # flatten (C*H*W) por muestra
        a_flat = a.reshape(N, -1)
        b_flat = b.reshape(N, -1)
        # normalizar en la dimensión vector (p=2)
        a_norm = F.normalize(a_flat, p=2, dim=1)
        b_norm = F.normalize(b_flat, p=2, dim=1)
        cos = (a_norm * b_norm).sum(dim=1)  # [N]
        return torch.mean(torch.abs(cos))   # promedio absoluto sobre batch

    sim23 = cos_per_sample(logits2, logits3)
    sim24 = cos_per_sample(logits2, logits4)
    sim34 = cos_per_sample(logits3, logits4)

    loss_me = (sim23 + sim24 + sim34) / 3.0
    return loss_me

# ============================================================
# 📗 PÉRDIDAS SEMI-SUPERVISADAS (para imágenes sin etiquetas)
# ============================================================

def unlabel_CE_loss2v1(logits2, prob3, prob4, th=0.8, criterion_cls=None):
    """
    Pseudo-labeling para rama 2:
    - pseudo_target = (prob3 + prob4)/2  (probabilidades por pixel)
    - conf_mask: por-pixel confianza máxima > th (shape [N,1,H,W])
    - cross-entropy contra p2 (soft targets) utilizando entropía cruzada para soft labels
    Retorna: (loss_scalar, pseudo_target_detached)
    """
    device = logits2.device
    p2 = F.softmax(logits2, dim=1)  # [N, C, H, W]
    pseudo_target = ((prob3 + prob4) / 2.0).detach()  # [N, C, H, W]

    # confidence por pixel (max over classes)
    conf_vals = pseudo_target.max(dim=1, keepdim=True)[0]  # [N,1,H,W]
    conf_mask = (conf_vals > th).float()  # 1 donde confiable

    # soft cross-entropy: - sum( q * log p )
    # produce shape [N,1,H,W]
    sce = -(pseudo_target * torch.log(p2 + EPS)).sum(dim=1, keepdim=True)

    # aplicar máscara y promediar sólo sobre píxeles confiables (evitar dividir por 0)
    masked = sce * conf_mask
    if conf_mask.sum() > 0:
        loss = masked.sum() / (conf_mask.sum() + EPS)
    else:
        # si no hay píxeles confiables, pérdida = 0 (evita NaNs)
        loss = torch.tensor(0.0, device=device)

    return loss, pseudo_target


def unlabel_CE_loss3v1(logits3, prob2, prob4, th=0.8, criterion_cls=None):
    p3 = F.softmax(logits3, dim=1)
    pseudo_target = ((prob2 + prob4) / 2.0).detach()
    conf_vals = pseudo_target.max(dim=1, keepdim=True)[0]
    conf_mask = (conf_vals > th).float()
    sce = -(pseudo_target * torch.log(p3 + EPS)).sum(dim=1, keepdim=True)
    masked = sce * conf_mask
    if conf_mask.sum() > 0:
        loss = masked.sum() / (conf_mask.sum() + EPS)
    else:
        loss = torch.tensor(0.0, device=logits3.device)
    return loss, pseudo_target


def unlabel_CE_loss4v1(logits4, prob2, prob3, th=0.8, criterion_cls=None):
    p4 = F.softmax(logits4, dim=1)
    pseudo_target = ((prob2 + prob3) / 2.0).detach()
    conf_vals = pseudo_target.max(dim=1, keepdim=True)[0]
    conf_mask = (conf_vals > th).float()
    sce = -(pseudo_target * torch.log(p4 + EPS)).sum(dim=1, keepdim=True)
    masked = sce * conf_mask
    if conf_mask.sum() > 0:
        loss = masked.sum() / (conf_mask.sum() + EPS)
    else:
        loss = torch.tensor(0.0, device=logits4.device)
    return loss, pseudo_target

# ============================================================
# 🧠 NOTAS:
# ============================================================
# - Los umbrales DENSITY_THRESHOLDS pueden modificarse para tu dataset.
#   Por ejemplo, si las cerezas de café están más separadas, aumenta "low" y "mid".
# - WEIGHTS define el peso de cada componente de la pérdida total.
# - Este módulo está diseñado para integrarse con el CSRNet modificado
#   dentro del framework CDMENet (3 ramas de densidad).
# ============================================================
