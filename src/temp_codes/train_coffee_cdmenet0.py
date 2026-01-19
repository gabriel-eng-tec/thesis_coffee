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
from model_cdmenet import CDMENet


# ============================================================
# ⚙️ CONFIGURACIÓN GLOBAL (fácil de modificar)
# ============================================================
dataset_dir = "../dataset/coffee_Fruit_Maturity_yolo"
TRAIN_JSON = f"{dataset_dir}/train/cdmenet_coffee_train.json"
VAL_JSON = f"{dataset_dir}/valid/cdmenet_coffee_val.json"
OUTPUT_DIR = "../models/cdmenet/checkpoints"

BATCH_SIZE = 1
LEARNING_RATE = 1e-5
EPOCHS = 1#200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pesos de pérdidas
WEIGHT_DENSITY = 1.0
WEIGHT_AUX = 0.1
WEIGHT_MUTUAL = 0.01

SAVE_INTERVAL = 10  # guardar cada N epochs
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
# 🔢 PÉRDIDAS
# ============================================================
def mutual_exclusion_loss(aux_outputs):
    """Exclusión mutua entre ramas auxiliares de CDMENet"""
    loss = 0
    for i in range(len(aux_outputs)):
        for j in range(i + 1, len(aux_outputs)):
            pi = torch.softmax(aux_outputs[i], dim=1)
            pj = torch.softmax(aux_outputs[j], dim=1)
            loss += (pi * pj).sum(dim=1).mean()
    return loss


# ============================================================
# 🚀 ENTRENAMIENTO
# ============================================================
def train_epoch(model, dataloader, optimizer, criterion_density, criterion_aux):
    model.train()
    epoch_loss = 0.0

    for imgs, dens in dataloader:
        imgs = imgs.to(DEVICE)
        dens = dens.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)

        density_pred = outputs[0]
        aux_outputs = outputs[1:]

        # --- Pérdidas ---
        loss_density = criterion_density(density_pred, dens)
        loss_aux = sum(criterion_aux(aux, torch.argmax(dens > 0, dim=1)) for aux in aux_outputs)
        loss_mutual = mutual_exclusion_loss(aux_outputs)

        loss = (WEIGHT_DENSITY * loss_density +
                WEIGHT_AUX * loss_aux +
                WEIGHT_MUTUAL * loss_mutual)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion_density):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, dens in dataloader:
            imgs = imgs.to(DEVICE)
            dens = dens.to(DEVICE)
            density_pred = model(imgs)[0]
            loss = criterion_density(density_pred, dens)
            val_loss += loss.item()
    return val_loss / len(dataloader)


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
    criterion_density = nn.MSELoss()
    criterion_aux = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion_density, criterion_aux)
        val_loss = validate_epoch(model, val_loader, criterion_density)
        elapsed = time.time() - start

        print(f"[{epoch:03d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Tiempo: {elapsed:.1f}s")

        # Guardar modelo
        if epoch % SAVE_INTERVAL == 0 or val_loss < best_val_loss:
            best_val_loss = min(best_val_loss, val_loss)
            save_path = os.path.join(OUTPUT_DIR, f"cdmenet_epoch{epoch:03d}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Modelo guardado: {save_path}")


if __name__ == "__main__":
    main()
