"""
train_yolo_cdmenet_style_full.py — Entrenamiento tipo CDMENet usando YOLOv8
Autor: Gabriel Barboza Álvarez
Compatibilidad: PyTorch 2.x / Ultralytics ≥8.0
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.yolo.utils import LOGGER

# ============================================================
# ⚙️ CONFIGURACIÓN GLOBAL
# ============================================================
DATASET_DIR = "../dataset/coffee_Fruit_Maturity_yolo"
DATA_YAML = f"{DATASET_DIR}/data.yaml"
OUTPUT_DIR = "../models/yolov8m_cdmenet_style"

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hiperparámetros
IMAGE_SIZE = 1024
EPOCHS_TOTAL = 100
EPOCH_WARMUP = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BLOCK_SIZE = 20  # guardar checkpoint cada 20 epochs

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "yolo_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "yolo_best.pth")

# ============================================================
# 🧱 UTILIDAD
# ============================================================
def freeze_backbone(model, freeze=True):
    for p in model.model[0].parameters():
        p.requires_grad = not freeze
    print("✅ Backbone congelado" if freeze else "🔓 Backbone desbloqueado")

def limpiar_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============================================================
# 🚀 ENTRENAMIENTO
# ============================================================
def train_epoch(model, dataloader, optimizer, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Entrenando ep {epoch}/{total_epochs}", unit="batch", dynamic_ncols=True)

    for imgs, labels in progress_bar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model.model(imgs, labels)  # forward nativo YOLO
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, len(dataloader))

@torch.no_grad()
def validate_epoch(model, dataloader, epoch, total_epochs):
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    n_images = 0

    progress_bar = tqdm(dataloader, desc=f"Validando ep {epoch}/{total_epochs}", unit="batch", dynamic_ncols=True)

    for imgs, labels in progress_bar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        outputs = model.model(imgs, labels)
        loss = outputs["loss"].item()
        total_loss += loss

        # Conteo: detecciones vs GT
        preds = model.model.predictor.model(imgs)
        for i, p in enumerate(preds):
            pred_count = len(p.boxes)
            gt_count = len(labels[i])
            err = abs(pred_count - gt_count)
            total_abs_error += err
            total_sq_error += err ** 2
            n_images += 1

        progress_bar.set_postfix({"loss": f"{loss:.4f}"})

    mae = total_abs_error / max(1, n_images)
    rmse = np.sqrt(total_sq_error / max(1, n_images))
    return total_loss / max(1, len(dataloader)), mae, rmse

# ============================================================
# 💾 CHECKPOINTS
# ============================================================
def save_checkpoint(model, optimizer, epoch, best_val, path):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val
    }
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint guardado en {path}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    best_val = checkpoint.get("best_val_loss", float("inf"))
    print(f"✅ Checkpoint cargado desde {path}, reanudando en epoch {start_epoch}")
    return model, optimizer, start_epoch, best_val

# ============================================================
# 🏁 MAIN
# ============================================================
def main():
    print("=== Entrenamiento YOLO estilo CDMENet ===")
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Crear modelo YOLO desde pesos base
    model = YOLO(f"../models/yolov8m/yolov8m-seg.pt").to(DEVICE)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Dataset con método interno de YOLO (para compatibilidad rápida)
    train_loader = model.build_dataloader(DATA_YAML, imgsz=IMAGE_SIZE, batch=BATCH_SIZE, mode="train")
    val_loader = model.build_dataloader(DATA_YAML, imgsz=IMAGE_SIZE, batch=BATCH_SIZE, mode="val")

    start_epoch = 1
    best_val_loss = float("inf")

    if os.path.exists(CHECKPOINT_PATH):
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, CHECKPOINT_PATH)

    freeze_backbone(model, True)

    for epoch in range(start_epoch, EPOCHS_TOTAL + 1):
        t0 = time.time()

        # Desbloquear backbone y ajustar LR al terminar warm-up
        if epoch == EPOCH_WARMUP:
            freeze_backbone(model, False)
            for g in optimizer.param_groups:
                g['lr'] *= 0.5
            print(f"🔓 Backbone desbloqueado. LR reducido a {optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_epoch(model, train_loader, optimizer, epoch, EPOCHS_TOTAL)
        val_loss, mae, rmse = validate_epoch(model, val_loader, epoch, EPOCHS_TOTAL)

        elapsed = (time.time() - t0) / 60.0
        print(f"[{epoch:03d}/{EPOCHS_TOTAL}] TrainLoss={train_loss:.4f} | "
              f"ValLoss={val_loss:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f} | Tiempo={elapsed:.2f}min")

        # Guardar checkpoint cada 20 epochs
        if epoch % BLOCK_SIZE == 0:
            save_checkpoint(model, optimizer, epoch, best_val_loss, CHECKPOINT_PATH)

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.model.state_dict(), BEST_MODEL_PATH)
            print(f"🥇 Nuevo mejor modelo guardado en {BEST_MODEL_PATH}")

        limpiar_gpu()

        # Pausa de enfriamiento
        if epoch % 2 == 0:
            print("🕒 Esperando 3 min para enfriar GPU...")
            time.sleep(180)

    print("✅ Entrenamiento completado.")

if __name__ == "__main__":
    main()
