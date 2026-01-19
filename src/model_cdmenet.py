"""
model_cdmenet.py — CDMENet adaptado para conteo de bayas de café
Objetivo: conteo total, sin clasificación por madurez
Basado en: Tang et al., CDMENet (2021)
Compatibilidad: PyTorch 2.x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_BN_Weights

# ============================================================
# ⚙️ CONFIGURACIÓN GLOBAL
# ============================================================
BACKBONE_PRETRAINED = True      # True: usa pesos VGG16 preentrenados
BACKBONE_NAME = "vgg16_bn"      # Puedes cambiarlo fácilmente
DENSITY_CHANNELS = 1            # Salida: mapa de densidad
AUX_BRANCHES = 3                # Ramas auxiliares de CDMENet
AUX_CHANNELS = 2                # Clases de densidad (fondo / alta densidad)

# ============================================================
# 🧱 FUNCIONES DE CONSTRUCCIÓN
# ============================================================
def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """Crea capas convolucionales tipo VGG."""
    layers = []
    d_rate = 2 if dilation else 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# ============================================================
# 🧩 MODELO CDMENet
# ============================================================
class CDMENet(nn.Module):
    def __init__(self, image_size: int = 512,
                 stride8: bool | None = None,
                 out_relu: bool | None = None,
                 upsample_to_input: bool | None = None):
        super(CDMENet, self).__init__()

        # --- Flags locales (conservan tus defaults y sólo ajustan si image_size != 512 o si se pasan explícitamente) ---
        self.image_size = image_size
        self._stride8 = True if stride8 is None else bool(stride8)
        self._out_relu = True if out_relu is None else bool(out_relu)
        self._upsample_to_input = False if upsample_to_input is None else bool(upsample_to_input)
        if self.image_size != 512:
            # Ajustes razonables por defecto para resoluciones distintas a 512
            if stride8 is None:
                self._stride8 = True
            if out_relu is None:
                self._out_relu = True
            # Mantén upsample apagado por defecto (visualización opcional)
            if upsample_to_input is None:
                self._upsample_to_input = False

        # --- 1️⃣ BACKBONE (VGG16) ---
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if BACKBONE_PRETRAINED else None
        vgg = models.vgg16_bn(weights=weights)
        features = list(vgg.features.children())
        out = []
        pool_count = 0

        for m in features:
            if isinstance(m, nn.MaxPool2d):
                pool_count += 1
                # Omitir el 4º y 5º MaxPool (igual que el paper original)
                if self._stride8 and pool_count >= 4:
                    continue
            out.append(m)

        self.frontend_feat = nn.Sequential(*out)

        # --- 2️⃣ BACKEND (CSRNet dilated convs) ---
        self.backend_feat = make_layers(
            [512, 512, 512, 256, 128, 64],
            in_channels=512,
            dilation=True,
            batch_norm=False
        )

        # --- 3️⃣ SALIDA PRINCIPAL ---
        self.output_layer = nn.Conv2d(64, DENSITY_CHANNELS, kernel_size=1)
        self.out_act = nn.ReLU(inplace=True) if self._out_relu else nn.Identity()

        # --- 4️⃣ RAMAS AUXILIARES (CDMENet) ---
        self.aux_layers = nn.ModuleList([
            nn.Conv2d(64, AUX_CHANNELS, kernel_size=1)
            for _ in range(AUX_BRANCHES)
        ])

        # 5) Upsampling opcional a tamaño de entrada
        self.upsample = nn.Upsample(mode="bilinear", align_corners=False)

        self._curr_epoch = 0

        # --- 5️⃣ Inicialización ---
        self._initialize_weights()

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.frontend_feat(x)
        x = self.backend_feat(x)
        density_map = self.out_act(self.output_layer(x))
        aux_outputs = [aux(x) for aux in self.aux_layers]

        if self._upsample_to_input:
            # Upsample dinámico: usar F.interpolate para pasar size en runtime
            density_map = F.interpolate(density_map, size=(H, W), mode="bilinear", align_corners=False)

        return (density_map, *aux_outputs)

    def _initialize_weights(self):
        for m in self.backend_feat.children():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0)
        for aux in self.aux_layers:
            nn.init.normal_(aux.weight, std=0.01)
            nn.init.constant_(aux.bias, 0)
