import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

# ==============================
# CONFIGURACIÓN
# ==============================
dataset_dir = "../dataset/coffee_Fruit_Maturity_yolo/train"
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")
density_dir = os.path.join(dataset_dir, "density_labels")

os.makedirs(density_dir, exist_ok=True)

sigma = 7        # nivel de suavizado base (radio gaussiano)
boost_factor = 1.5  # cuánto amplificar la densidad en zonas agrupadas
neighbor_radius = 50  # píxeles para considerar "agrupadas"

# ==============================
# FUNCIÓN PARA INFERIR DENSIDAD BAJO OCLUSIÓN
# ==============================
def make_inferred_density(img_shape, points, sigma=7, boost_factor=1.5, neighbor_radius=50):
    """Crea un mapa de densidad que aumenta el valor en zonas donde hay agrupamientos,
       simulando bayas ocultas o parcialmente visibles."""
    h, w = img_shape[:2]
    base = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return base

    # 1️⃣ Colocar puntos base
    for (x, y) in points:
        if 0 <= x < w and 0 <= y < h:
            base[int(y), int(x)] += 1

    # 2️⃣ Calcular agrupamiento (densidad local)
    nbrs = NearestNeighbors(radius=neighbor_radius).fit(points)
    densities = np.array([len(nbrs.radius_neighbors([p], return_distance=False)[0]) for p in points])

    # 3️⃣ Aumentar "peso" en zonas densas
    for (x, y), local_density in zip(points, densities):
        weight = 1 + boost_factor * (local_density / max(1, len(points)))  # amplificación local
        if 0 <= x < w and 0 <= y < h:
            base[int(y), int(x)] += weight

    # 4️⃣ Suavizar con filtro gaussiano
    inferred_density = gaussian_filter(base, sigma=sigma)

    # 5️⃣ Normalizar la suma total al número de objetos (para mantener coherencia)
    sum_before = np.sum(base)
    sum_after = np.sum(inferred_density)
    if sum_after > 0:
        inferred_density *= sum_before / sum_after

    return inferred_density

# ==============================
# FUNCIÓN PARA PROCESAR UN ARCHIVO YOLO
# ==============================
def generate_density_from_label(image_path, label_path, sigma=7):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ No se pudo leer la imagen: {image_path}")
        return None, None

    h, w = img.shape[:2]
    points = []

    # Leer archivo de etiquetas YOLO-seg
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            cls_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))

            # Si hay pares de coordenadas (x1, y1, x2, y2, ...)
            if len(coords) % 2 == 0:
                pts = np.array([[coords[i] * w, coords[i + 1] * h] for i in range(0, len(coords), 2)])
                cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                points.append((int(cx), int(cy)))
            else:
                # Si por alguna razón hay un formato raro, lo ignoramos
                continue

    # Generar mapa inferido con atención a agrupamientos
    density_map = make_inferred_density((h, w), points, sigma=sigma,
                                        boost_factor=boost_factor,
                                        neighbor_radius=neighbor_radius)
    return density_map, img

# ==============================
# PROCESAR TODO EL DATASET
# ==============================
for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    img_name = label_file.replace(".txt", ".jpg")
    img_path = os.path.join(images_dir, img_name)
    label_path = os.path.join(labels_dir, label_file)

    if not os.path.exists(img_path):
        print(f"⚠️ Imagen no encontrada para {label_file}")
        continue

    density_map, original_img = generate_density_from_label(img_path, label_path, sigma=sigma)
    if density_map is None:
        print(f"⚠️ Density map is None para {label_file}")
        continue

    # Guardar .npy
    npy_path = os.path.join(density_dir, label_file.replace(".txt", ".npy"))
    np.save(npy_path, density_map)

    # Crear vista previa
    density_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    density_norm = density_norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(original_img, 0.6, heatmap, 0.7, 0)

    preview_path = os.path.join(density_dir, label_file.replace(".txt", "_heatmap.jpg"))
    cv2.imwrite(preview_path, blended)

    print(f"✅ Guardado: {npy_path}")
    print(f"🖼️ Preview: {preview_path}")

print("\n🎯 Todos los mapas de densidad inferidos fueron generados en:")
print(f"➡️ {density_dir}")
