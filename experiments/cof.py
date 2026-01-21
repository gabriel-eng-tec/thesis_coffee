import os
import torch.optim as optim
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import json
import h5py

from model_cdmenet import CDMENet

# ------------------ CONFIG ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dataset_dir = "../dataset/coffee_Fruit_Maturity_yolo"
VAL_JSON = f"{dataset_dir}/valid/cdmenet_coffee_valid.json"
CHECKPOINT_PATH = "../models/cdmenet/checkpoints/cdmenet_checkpoint.pth"
LEARNING_RATE = 1e-5

# ------------------ DATASET ------------------
class CoffeeDensityDataset(Dataset):
    """Carga imágenes y mapas de densidad .h5 desde JSON"""
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
        return Image.open(path).convert("RGB")

    def _load_density(self, path):
        with h5py.File(path, 'r') as f:
            density = np.array(f['density'])
        return density.astype(np.float32)

# ------------------ FUNCIONES ------------------
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"✅ Checkpoint cargado desde {path}, epoch={epoch}")
    return model, optimizer, epoch, best_val_loss


def validate_epoch(model, dataloader):
    model.eval()
    total_mae = 0.0
    total_rmse = 0.0

    all_preds, all_gts = [], []

    progress_bar = tqdm(dataloader, desc="Validando", unit="batch", dynamic_ncols=True)

    with torch.no_grad():
        for imgs, dens in progress_bar:
            imgs = imgs.to(DEVICE)
            dens = dens.to(DEVICE)

            density_pred = model(imgs)[0]
            dens_resized = F.interpolate(dens, size=density_pred.shape[2:], mode='bilinear', align_corners=False)

            # Conteos
            pred_count = density_pred.sum().item()
            gt_count = dens_resized.sum().item()

            all_preds.append(pred_count)
            all_gts.append(gt_count)

            mae = abs(pred_count - gt_count)
            rmse = (pred_count - gt_count) ** 2

            total_mae += mae
            total_rmse += rmse

            progress_bar.set_postfix({
                "MAE": f"{total_mae / (progress_bar.n + 1):.2f}",
                "GT": f"{gt_count:.1f}",
                "Pred": f"{pred_count:.1f}"
            })

    # Estadísticas globales
    mean_gt = np.mean(all_gts)
    mean_pred = np.mean(all_preds)
    scale_ratio = mean_gt / (mean_pred + 1e-8)

    print("\n📊 === Estadísticas de Validación ===")
    print(f"MAE Promedio: {total_mae / len(dataloader):.3f}")
    print(f"RMSE Promedio: {(total_rmse / len(dataloader))**0.5:.3f}")
    print(f"Conteo medio real (GT): {mean_gt:.2f}")
    print(f"Conteo medio predicho: {mean_pred:.2f}")
    print(f"Factor de corrección sugerido: x{scale_ratio:.3f}")

    if scale_ratio < 0.5 or scale_ratio > 1.5:
        print("⚠️  Los mapas de densidad parecen no estar correctamente normalizados.")
        print("   Revisa si la suma de cada mapa de densidad ≈ número de frutos en la anotación.")

    return total_mae / len(dataloader)


# ------------------ PIPELINE ------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

val_set = CoffeeDensityDataset(VAL_JSON, transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

model = CDMENet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model, _, _, _ = load_checkpoint(model, optimizer, CHECKPOINT_PATH)
val_loss = validate_epoch(model, val_loader)

print(f"\n✅ Validación finalizada. MAE total: {val_loss:.3f}")

"""
import h5py, numpy as np, json

val = "valid"
dataset_json = f"../dataset/coffee_Fruit_Maturity_yolo/{val}/cdmenet_coffee_{val}.json"
with open(dataset_json, "r") as f:
    data = json.load(f)

total_sum = 0
for item in data:
    with h5py.File(item["density"], "r") as hf:
        den = np.array(hf["density"])
        total_sum += den.sum()

print("Promedio de suma de densidades:", total_sum / len(data))
"""
