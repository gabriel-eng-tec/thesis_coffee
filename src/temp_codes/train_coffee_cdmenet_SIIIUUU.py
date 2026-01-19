"""
train_cdmenet.py — Entrenamiento del modelo CDMENet para conteo de bayas de café
Basado en: Tang et al., CDMENet (2021)
Autor original: Gabriel Barboza Álvarez
Edición: Ajustes de warm-up, freeze/unfreeze, y logging (512x512)
Compatibilidad: PyTorch 2.x
"""

import os
import sys
import json
import time
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model_cdmenet import CDMENet
from util_mutual_exec import (
    densitymap_to_densitymask,
    unlabel_CE_loss2v1, unlabel_CE_loss3v1, unlabel_CE_loss4v1,
    mutual_exclusion_loss, cross_entropy_loss,
)

# ============================================================
# ⚙️ CONFIGURACIÓN GLOBAL
# ============================================================
dataset_dir = "../dataset/coffee_Fruit_Maturity_yolo"
# Si quieres entrenar rápido con valid/test:
TRAIN_JSON = f"{dataset_dir}/valid/cdmenet_coffee_valid.json"
VAL_JSON   = f"{dataset_dir}/test/cdmenet_coffee_test.json"
OUTPUT_DIR = "../models/cdmenet/checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibilidad
rand_seed = 123456
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(rand_seed)

# Hiperparámetros
IMAGE_SIZE     = 512       # <— Alineado con tu dataset/GT
LEARNING_RATE  = 1e-4      # un poco más alto para el arranque
BATCH_SIZE     = 4
EPOCHS_TOTAL   = 20        # ajusta a gusto
BLOCK_SIZE     = 5         # imprime en bloques
WEIGHT_DECAY   = 1e-4

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "cdmenet_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "cdmenet_epoch_best.pth")

# ============================================================
# 🧱 DATASET
# ============================================================
class CoffeeDensityDataset(Dataset):
    """Dataset que carga imágenes y mapas de densidad .h5 desde JSON"""
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self._load_image(item['image'])
        dens = None
        if item.get("label", True):
            dens = self._load_density(item['density'])
        if self.transform:
            img = self.transform(img)
        if dens is not None:
            dens = torch.from_numpy(dens).unsqueeze(0)  # [1,H,W]
        return img, dens

    def _load_image(self, path):
        from PIL import Image, ImageOps
        # exif_transpose para evitar giros inesperados
        img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
        return img

    def _load_density(self, path):
        with h5py.File(path, 'r') as f:
            # dataset 'density' esperado
            density = np.array(f['density'])
        return density.astype(np.float32)

# ============================================================
# 🧱 UTILIDAD
# ============================================================
def limpiar_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def freeze_frontend(model, freeze: bool):
    if not hasattr(model, "frontend_feat"):
        return
    for p in model.frontend_feat.parameters():
        p.requires_grad = not (not not freeze)  # cast robusto
    # Nota: no llamamos .eval() para BN; mantenemos train() en el loop

# ============================================================
# 🚀 ENTRENAMIENTO
# ============================================================
def train_epoch(model, dataloader, optimizer, epoch, total_epochs):
    model.train()
    # Señal para decidir warm-up dentro del forward de pérdidas
    model._curr_epoch = epoch

    # Warm-up: solo MSE + conteo en 1–3; luego auxiliares suaves
    use_aux = epoch >= 4
    w_class = 0.005 if use_aux else 0.0
    w_me    = 0.3   if use_aux else 0.0
    w_semi  = 0.0   # hasta que realmente metas label:0

    running = {"ld": 0.0, "lc": 0.0, "lcls": 0.0, "lme": 0.0, "n": 0}
    progress_bar = tqdm(dataloader, desc=f"Entrenando ep {epoch}/{total_epochs}", unit="batch", dynamic_ncols=True)

    for bidx, (imgs, dens) in enumerate(progress_bar):
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()

        # ¿Tiene GT de densidad?
        has_label = dens is not None
        if has_label:
            dens = dens.to(DEVICE)

        density_pred, logits2, logits3, logits4 = model(imgs)
        

        loss = 0.0
        loss_density = torch.tensor(0.0, device=DEVICE)
        loss_count   = torch.tensor(0.0, device=DEVICE)
        loss_cls_avg = torch.tensor(0.0, device=DEVICE)
        loss_me      = torch.tensor(0.0, device=DEVICE)

        if has_label:
            # Resize GT a la salida del modelo (suele ser menor por el stride)
            dens_resized = F.interpolate(dens, size=density_pred.shape[2:], mode='bilinear', align_corners=False)

            # MSE de densidad
            loss_density = F.mse_loss(density_pred, dens_resized)

            # Loss de conteo (ayuda a “despegar” de pred=0)
            pred_cnt = density_pred.sum(dim=[1,2,3])
            gt_cnt   = dens_resized.sum(dim=[1,2,3])
            loss_count = 0.1 * torch.mean(torch.abs(pred_cnt - gt_cnt))

            # Auxiliares (desde ep4)
            if use_aux:
                t2 = densitymap_to_densitymask(dens_resized, 0.0, 0.0015).squeeze(1).long()   # low
                t3 = densitymap_to_densitymask(dens_resized, 0.0015, 0.0100).squeeze(1).long() # mid
                t4 = densitymap_to_densitymask(dens_resized, 0.0100, 1.0).squeeze(1).long()     # high

                loss_aux2 = cross_entropy_loss(logits2, t2)
                loss_aux3 = cross_entropy_loss(logits3, t3)
                loss_aux4 = cross_entropy_loss(logits4, t4)
                loss_cls_avg = (loss_aux2 + loss_aux3 + loss_aux4) / 3.0

            loss = (1.0 * loss_density) + loss_count + (w_class * loss_cls_avg)
        else:
            # Semi-supervisado (apagado en warm-up)
            if w_semi > 0.0:
                prob2 = F.softmax(logits2, dim=1)
                prob3 = F.softmax(logits3, dim=1)
                prob4 = F.softmax(logits4, dim=1)
                l2, _ = unlabel_CE_loss2v1(logits2, prob3, prob4, th=0.8)
                l3, _ = unlabel_CE_loss3v1(logits3, prob2, prob4, th=0.8)
                l4, _ = unlabel_CE_loss4v1(logits4, prob2, prob3, th=0.8)
                loss += w_semi * (l2 + l3 + l4) / 3.0

        # Exclusión mutua (suave)
        loss_me = mutual_exclusion_loss(logits2, logits3, logits4)
        loss += w_me * loss_me

        # Backprop
        loss.backward()
        optimizer.step()

        # Métricas rápidas
        running["ld"] += float(loss_density.item())
        running["lc"] += float(loss_count.item())
        running["lcls"] += float(loss_cls_avg.item())
        running["lme"] += float(loss_me.item())
        running["n"] += 1

        # Logging compacto
        if bidx == 0 and has_label:
            print(f"[ep {epoch}] gt_mean={gt_cnt.mean().item():.2f} pred_mean={pred_cnt.mean().item():.2f} | "
                  f"Ldens={loss_density.item():.4f} Lcnt={loss_count.item():.4f} "
                  f"Lcls={loss_cls_avg.item():.4f} Lme={loss_me.item():.4f}")

        avg_loss = (running["ld"] + running["lc"] + running["lcls"] + running["lme"]) / max(running["n"], 1)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    avg_ld   = running["ld"]   / max(running["n"], 1)
    avg_lcnt = running["lc"]   / max(running["n"], 1)
    avg_lcls = running["lcls"] / max(running["n"], 1)
    avg_lme  = running["lme"]  / max(running["n"], 1)
    return avg_ld + avg_lcnt + avg_lcls + avg_lme


def validate_epoch(model, dataloader, epoch, total_epochs):
    model.eval()
    total_mae = 0.0
    progress_bar = tqdm(dataloader, desc=f"Validando ep {epoch}/{total_epochs}", unit="batch", dynamic_ncols=True)

    with torch.no_grad():
        for imgs, dens in progress_bar:
            imgs = imgs.to(DEVICE)
            dens = dens.to(DEVICE)

            density_pred = model(imgs)[0]
            dens_resized = F.interpolate(dens, size=density_pred.shape[2:], mode='bilinear', align_corners=False)

            mae = torch.abs(density_pred.sum(dim=[1,2,3]) - dens_resized.sum(dim=[1,2,3])).mean().item()
            total_mae += mae

            # Actualizar barra con MAE promedio
            progress_bar.set_postfix({"MAE": f"{total_mae / (progress_bar.n + 1):.2f}"})

    return total_mae / len(dataloader)

# ============================================================
# 🚀 FUNCIONES AUXILIARES
# ============================================================
def save_checkpoint(model, optimizer, epoch, best_val_loss, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint guardado en {path}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"✅ Checkpoint cargado desde {path}, comenzando en epoch {epoch+1}")
    return model, optimizer, epoch, best_val_loss

# ============================================================
# 🏁 MAIN
# ============================================================
def main():
    print("=== Entrenamiento CDMENet para conteo de bayas ===")

    if DEVICE != "cuda":
        resp = input("No se detectó GPU. ¿Desea continuar en CPU? (s/n): ").strip().lower()
        if resp != "s":
            print("Saliendo del programa...")
            sys.exit(0)
        else:
            print("Continuando en CPU...")

    # Transformaciones (alineadas a 512x512)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4931, 0.5346, 0.3792],
                             std=[0.2217, 0.2025, 0.2085])
    ])

    # Datasets y DataLoaders
    train_set = CoffeeDensityDataset(TRAIN_JSON, transform)
    val_set = CoffeeDensityDataset(VAL_JSON, transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Modelo y optimizer
    model = CDMENet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch = 1
    best_val_loss = float("inf")

    # Cargar checkpoint si existe
    if os.path.exists(CHECKPOINT_PATH):
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, CHECKPOINT_PATH)
        start_epoch += 1

    # Freeze del frontend durante warm-up (épocas 1–2)
    freeze_frontend(model, True)

    # Entrenamiento por bloques
    while start_epoch <= (EPOCHS_TOTAL + 1):
        end_epoch = min(start_epoch + BLOCK_SIZE - 1, EPOCHS_TOTAL)
        print(f"\n🚀 Entrenando epochs {start_epoch} a {end_epoch} de {EPOCHS_TOTAL}\n")

        for epoch in range(start_epoch, end_epoch + 1):
            t0 = time.time()

            # Descongelar frontend al empezar época 3 y bajar LR ×0.5
            if epoch == 3:
                freeze_frontend(model, False)
                for g in optimizer.param_groups:
                    g['lr'] *= 0.5
                print("🔓 Frontend desbloqueado y LR reducido a", optimizer.param_groups[0]['lr'])

            train_loss = train_epoch(model, train_loader, optimizer, epoch, EPOCHS_TOTAL)
            val_loss   = validate_epoch(model, val_loader, epoch, EPOCHS_TOTAL)

            elapsed = (time.time() - t0) / 60.0
            print(f"[{epoch:03d}/{EPOCHS_TOTAL}] TrainLoss: {train_loss:.4f} | ValMAE: {val_loss:.4f} | Tiempo: {elapsed:.2f} min")

            # Guardar checkpoint cada epoch
            save_checkpoint(model, optimizer, epoch, best_val_loss, CHECKPOINT_PATH)

            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"🥇 Nuevo mejor modelo guardado en {BEST_MODEL_PATH}")

            limpiar_gpu()

            # Espera de 5 minutos cada 2 épocas
            if epoch % 2 == 0:
                print("🕒 Wait 5 minutes to cold the system before the next epoch...")
                time.sleep(2 * 60)

        # salir tras este bloque
        break

if __name__ == "__main__":
    main()
