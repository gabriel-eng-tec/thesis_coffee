import os
import glob
import cv2
import h5py
import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# --------------------------
# Parámetros configurables
# --------------------------
gen = "train"
dataset_dir = f"../dataset/coffee_Fruit_Maturity_yolo/{gen}"
IMG_FOLDER = f"{dataset_dir}/images"          # Carpeta con imágenes
LABEL_FOLDER = f"{dataset_dir}/labels"        # Carpeta con labels YOLO
OUTPUT_FOLDER = f"{dataset_dir}/density_h5"   # Carpeta donde se guardan los .h5 y PNG
SIGMA = 7                         # Sigma de la gaussiana para difuminar puntos
PREVIEW = False                   # True para generar preview de mapas sobre imagen
PREVIEW_N = 5                    # Número de imágenes para guardar preview
JSON_FILENAME = f"cdmenet_coffee_{gen}.json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------
# Función para convertir labels YOLO a puntos
# --------------------------
def yolo_to_points(label_path, img_shape):
    height, width = img_shape[:2]
    points = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            if len(parts) >= 3:
                # Formato YOLO: class cx cy w h
                cx, cy = parts[1], parts[2]
                x = cx * width
                y = cy * height
                points.append((x, y))
    return points

# --------------------------
# Función para generar mapa de densidad normalizado
# --------------------------
def generate_density_map(points, img_shape, sigma=SIGMA):
    h, w = img_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return density
    for x, y in points:
        if 0 <= int(y) < h and 0 <= int(x) < w:
            density[int(y), int(x)] = 1.0

    # Aplicar filtro gaussiano
    density = gaussian_filter(density, sigma=sigma, mode='constant')

    # ✅ Normalización: suma total = número de puntos
    s = density.sum()
    if s > 0:
        density = density / s * len(points)

    return density

# --------------------------
# Función para guardar preview
# --------------------------
def save_preview(img_path, density_map, save_path):
    img = cv2.imread(img_path)
    density_normalized = (density_map / density_map.max() * 255).astype(np.uint8)
    density_colored = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, density_colored, 0.3, 0)
    cv2.imwrite(save_path, overlay)

def save_density_image(density_map, save_path):
    if density_map.max() > 0:
        norm_map = (density_map / density_map.max() * 255).astype(np.uint8)
    else:
        norm_map = density_map.astype(np.uint8)
    cv2.imwrite(save_path, norm_map)

# --------------------------
# Main
# --------------------------
def main():
    label_files = sorted(glob.glob(os.path.join(LABEL_FOLDER, "*.txt")))
    print(f"Encontrados {len(label_files)} archivos de etiquetas YOLO.")

    json_list = []

    for i, label_file in enumerate(tqdm(label_files)):
        img_file = os.path.join(IMG_FOLDER, os.path.basename(label_file).replace(".txt", ".jpg"))
        if not os.path.exists(img_file):
            continue

        img = cv2.imread(img_file)
        points = yolo_to_points(label_file, img.shape)
        density_map = generate_density_map(points, img.shape)

        # Guardar .h5
        h5_file = os.path.join(OUTPUT_FOLDER, os.path.basename(label_file).replace(".txt", ".h5"))
        with h5py.File(h5_file, 'w') as hf:
            hf.create_dataset('density', data=density_map)

        # Guardar mapa de densidad simple como PNG
        png_file = os.path.join(OUTPUT_FOLDER, os.path.basename(label_file).replace(".txt", ".png"))
        save_density_image(density_map, png_file)

        if PREVIEW and i < PREVIEW_N:
            preview_file = os.path.join(OUTPUT_FOLDER, os.path.basename(label_file).replace(".txt", "_preview.png"))
            save_preview(img_file, density_map, preview_file)

        json_list.append({
            "image": img_file,
            "density": h5_file,
            "label": 1
        })

        # ✅ Depuración: mostrar suma y número de puntos
        print(f"{os.path.basename(label_file)} -> {len(points)} puntos, suma densidad={density_map.sum():.2f}")

    # Guardar JSON completo
    json_path = os.path.join(dataset_dir, JSON_FILENAME)
    with open(json_path, "w") as f:
        json.dump(json_list, f, indent=4)

    print(f"\n✅ Mapas de densidad generados y JSON guardado en {json_path}")
    print(f"📁 Carpeta de salida: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
