"""
eval_cdmenet_full_v4.py — Evaluación CDMENet con overlay + conteo sobre imagen original
Autor: Gabriel Barboza Álvarez
Compatibilidad: PyTorch 2.x
"""

import os
import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.cm as cm
from model_cdmenet import CDMENet

# ------------------- CONFIGURACIÓN -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dataset_dir = "../dataset/coffee_Fruit_Maturity_yolo"

MODEL_PATH = "../models/cdmenet/checkpoints/cdmenet_epoch_best.pth"
#IMAGE_DIR = "../dataset/icafe/images/val"
IMAGE_DIR = f"{dataset_dir}/valid/images"
OUTPUT_DIR = f"{dataset_dir}/results_cdmenet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(OUTPUT_DIR, "conteos.csv")

TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4931, 0.5346, 0.3792],
                         std=[0.2217, 0.2025, 0.2085]),
])

# ------------------- CARGAR MODELO -------------------
model = CDMENet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------------------- FUNCIONES AUXILIARES -------------------
def generate_density_map(img_tensor):
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(DEVICE))[0]  # [1,1,H,W]
        density_map = output.squeeze().cpu().numpy()
        density_map = np.maximum(density_map, 0)
    return density_map

def resize_density_to_original(density_map, original_size):
    """Redimensiona el mapa de densidad al tamaño original y ajusta la suma"""
    h_orig, w_orig = original_size
    density_tensor = torch.tensor(density_map).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(density_tensor, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
    
    # Escalar para mantener la suma total
    scale_factor = float(density_map.sum() / (resized.squeeze().sum().item() + 1e-6))
    return resized.squeeze().numpy() * scale_factor

def overlay_density_on_image(img_pil, density_map, alpha=0.5):
    """Genera overlay del heatmap sobre la imagen original"""
    img_np = np.array(img_pil)

    # Normalización robusta
    vmax = max(np.percentile(density_map, 99), 1e-6)
    density_norm = np.clip(density_map / vmax, 0, 1)
    cmap = (cm.jet(density_norm)[:, :, :3] * 255).astype(np.uint8)

    # Asegurar mismo tamaño que la imagen
    if cmap.shape[:2] != img_np.shape[:2]:
        cmap = np.array(Image.fromarray(cmap).resize((img_np.shape[1], img_np.shape[0])))

    overlay = (1 - alpha) * img_np + alpha * cmap
    overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    return overlay_img

def add_text_to_image(img_pil, text, position=(10, 10)):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, fill=(255, 255, 255), font=font)
    return img_pil

# ------------------- EVALUACIÓN -------------------
results = []

for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    try:
        # --- Cargar imagen original ---
        img_path = os.path.join(IMAGE_DIR, filename)
        img_pil = Image.open(img_path).convert("RGB")
        original_size = img_pil.size[::-1]  # (H, W)

        # --- Generar mapa en escala del modelo ---
        img_tensor = TRANSFORM(img_pil)
        density_map_512 = generate_density_map(img_tensor)

        # --- Filtrar ruido mínimo ---
        #density_map_512[density_map_512 < 1e-3] = 0

        # --- Redimensionar al tamaño original y conservar conteo ---
        density_map_orig = resize_density_to_original(density_map_512, original_size)
        count = density_map_orig.sum()
        results.append((filename, count))

        print(f"{filename}: Conteo ajustado = {count:.2f}")

        # --- Generar overlay con heatmap y conteo ---
        overlay_img = overlay_density_on_image(img_pil, density_map_orig)
        overlay_img = add_text_to_image(overlay_img, f"Conteo: {count:.2f}")
        overlay_img.save(os.path.join(OUTPUT_DIR, f"overlay_{filename}"))

        break

    except Exception as e:
        print(f"⚠️ Error procesando {filename}: {e}")

# ------------------- GUARDAR CSV -------------------
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Imagen", "Conteo"])
    for filename, count in results:
        writer.writerow([filename, f"{count:.2f}"])

print(f"✅ Resultados guardados en {OUTPUT_DIR} y {CSV_FILE}")
