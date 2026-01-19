"""
train_cdmenet.py — Entrenamiento del modelo CDMENet para conteo de bayas de café
Basado en: Tang et al., CDMENet (2021)
Autor: Gabriel Barboza Álvarez
Compatibilidad: PyTorch 2.x
"""

import os
import sys
import json
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
from model_cdmenet import CDMENet
from util_mutual_exec import (
    densitymap_to_densitymask,
    unlabel_CE_loss2v1, unlabel_CE_loss3v1, unlabel_CE_loss4v1,
    mutual_exclusion_loss, cross_entropy_loss, WEIGHTS, DENSITY_THRESHOLDS
)

# ============================================================
# ⚙️ CONFIGURACIÓN GLOBAL (fácil de modificar)
# ============================================================
dataset_dir = "../dataset/coffee_Fruit_Maturity_yolo"
TRAIN_JSON = f"{dataset_dir}/train/cdmenet_coffee_train.json"
VAL_JSON = f"{dataset_dir}/valid/cdmenet_coffee_val.json"
# TRAIN_JSON = f"{dataset_dir}/valid/cdmenet_coffee_val.json"
# VAL_JSON = f"{dataset_dir}/test/cdmenet_coffee_test.json"
OUTPUT_DIR = "../models/cdmenet/checkpoints"

BATCH_SIZE = 4
LEARNING_RATE = 1e-5
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pesos de pérdidas
WEIGHT_DENSITY = 1.0
WEIGHT_AUX = 0.1
WEIGHT_MUTUAL = 0.01

SAVE_INTERVAL = 3  # guardar cada N epochs
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        den = self._load_density(item['density'])
        if self.transform:
            img = self.transform(img)
        return img, torch.from_numpy(den).unsqueeze(0)

    def _load_image(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return img

    def _load_density(self, path):
        with h5py.File(path, 'r') as f:
            density = np.array(f['density'])
        return density.astype(np.float32)

# ============================================================
# 🧱 CLEAN
# ============================================================
def limpiar_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# 🚀 ENTRENAMIENTO
# ============================================================
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Entrenando", unit="batch", dynamic_ncols=True)

    for imgs, dens in progress_bar:
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()

        # --- Determinar si hay densidad anotada ---
        has_label = dens is not None
        if has_label:
            dens = dens.to(DEVICE)

        # --- Forward ---
        outputs = model(imgs)
        density_pred, logits2, logits3, logits4 = outputs

        loss = 0.0

        if has_label:
            # --- Pérdida de densidad ---
            dens_resized = F.interpolate(dens, size=density_pred.shape[2:], mode='bilinear', align_corners=False)
            loss_density = F.mse_loss(density_pred, dens_resized)

            # --- Pérdidas auxiliares ---
            target_mask_2 = densitymap_to_densitymask(dens_resized, 0.0, DENSITY_THRESHOLDS["low"])
            target_mask_3 = densitymap_to_densitymask(dens_resized, DENSITY_THRESHOLDS["low"], DENSITY_THRESHOLDS["mid"])
            target_mask_4 = densitymap_to_densitymask(dens_resized, DENSITY_THRESHOLDS["mid"], 1.0)

            target_mask_2 = target_mask_2.squeeze(1).long()  # de [N,1,H,W] a [N,H,W]
            target_mask_3 = target_mask_3.squeeze(1).long()
            target_mask_4 = target_mask_4.squeeze(1).long()

            loss_aux2 = cross_entropy_loss(logits2, target_mask_2)
            loss_aux3 = cross_entropy_loss(logits3, target_mask_3)
            loss_aux4 = cross_entropy_loss(logits4, target_mask_4)

            # --- Combinar pérdidas ---
            loss += (WEIGHTS["density_loss"] * loss_density +
                     WEIGHTS["class_loss"] * (loss_aux2 + loss_aux3 + loss_aux4)/3)

        else:
            # --- Pérdidas semi-supervisadas ---
            prob2 = F.softmax(logits2, dim=1)
            prob3 = F.softmax(logits3, dim=1)
            prob4 = F.softmax(logits4, dim=1)

            loss2, _ = unlabel_CE_loss2v1(logits2, prob3, prob4, th=0.8)
            loss3, _ = unlabel_CE_loss3v1(logits3, prob2, prob4, th=0.8)
            loss4, _ = unlabel_CE_loss4v1(logits4, prob2, prob3, th=0.8)

            loss += WEIGHTS["semi_supervised"] * (loss2 + loss3 + loss4)/3

        # --- Exclusión mutua siempre ---
        loss_me = mutual_exclusion_loss(logits2, logits3, logits4)
        loss += WEIGHTS["mutual_exc"] * loss_me

        # --- Backprop ---
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # --- Actualizar barra ---
        progress_bar.set_postfix({"loss": f"{total_loss / (progress_bar.n + 1):.4f}"})

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader):
    model.eval()
    total_mae = 0.0

    progress_bar = tqdm(dataloader, desc="Validando", unit="batch", dynamic_ncols=True)

    with torch.no_grad():
        for imgs, dens in progress_bar:
            imgs = imgs.to(DEVICE)
            dens = dens.to(DEVICE)

            density_pred = model(imgs)[0]
            dens_resized = F.interpolate(dens, size=density_pred.shape[2:], mode='bilinear', align_corners=False)

            total_mae += torch.abs(density_pred.sum() - dens_resized.sum()).item()

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
        # Preguntar al usuario si quiere continuar en CPU
        respuesta = input("No se detectó GPU. ¿Desea continuar en CPU? (s/n): ").strip().lower()
        if respuesta != "s":
            print("Saliendo del programa...")
            sys.exit(0)  # Termina el programa
        else:
            print("Continuando en CPU...")
    
    os.environ["MIOPEN_FIND_MODE"] = "1"
    os.environ["MIOPEN_ENABLE_CACHE"] = "1"
    os.environ["MIOPEN_USER_DB_PATH"] = "/tmp/miopen_cache"
    os.environ["MIOPEN_DISABLE_WARNINGS"] = "1"

    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Datasets y DataLoaders
    train_set = CoffeeDensityDataset(TRAIN_JSON, transform)
    val_set = CoffeeDensityDataset(VAL_JSON, transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Modelo
    model = CDMENet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate_epoch(model, val_loader)
        elapsed = time.time() - start

        elapsed_minutes = elapsed / 60

        print(f"[{epoch:03d}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Tiempo: {elapsed_minutes:.2f} min")

        # Guardar modelo cada SAVE_INTERVAL
        if epoch % SAVE_INTERVAL == 0:
            save_path = os.path.join(OUTPUT_DIR, f"cdmenet_epoch{epoch:03d}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Modelo guardado (intervalo): {save_path}")

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(OUTPUT_DIR, "cdmenet_epoch_best.pth")
            torch.save(model.state_dict(), best_path)

        limpiar_gpu()


if __name__ == "__main__":
    main()
