"""
utils package — Utilidades compartidas (pérdidas, preprocessing, etc.)
"""

from .density import (
    yolo_to_points,
    generate_density_map,
    save_preview,
    save_density_image,
    main as generate_density_main
)

from .cdmenet_mutual_exec import (
    densitymap_to_densitymask,
    dice_loss,
    cross_entropy_loss,
    mutual_exclusion_loss,
    unlabel_CE_loss2v1,
    unlabel_CE_loss3v1,
    unlabel_CE_loss4v1,
    save_checkpoint0,
    DENSITY_THRESHOLDS,
    WEIGHTS,
    EPS
)

__all__ = [
    "yolo_to_points",
    "generate_density_map",
    "save_preview",
    "save_density_image",
    "generate_density_main",
    "densitymap_to_densitymask",
    "dice_loss",
    "cross_entropy_loss",
    "mutual_exclusion_loss",
    "unlabel_CE_loss2v1",
    "unlabel_CE_loss3v1",
    "unlabel_CE_loss4v1",
    "save_checkpoint0",
    "DENSITY_THRESHOLDS",
    "WEIGHTS",
    "EPS"
]
