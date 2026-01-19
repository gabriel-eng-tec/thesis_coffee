import os
import cv2
import numpy as np

# ==============================
# CONFIGURACIÓN
# ==============================
# Carpetas del dataset
dataset_path = "../dataset/coffee_Fruit_Maturity_yolo/train"
images_dir = os.path.join(dataset_path, "images")
labels_dir = os.path.join(dataset_path, "labels")

# Colores aleatorios por clase
np.random.seed(42)
colors = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)  # hasta 50 clases

# Mostrar una ventana o guardar imágenes
save_output = True
output_dir = "./mask_visuals"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# FUNCIÓN PARA DIBUJAR MÁSCARAS
# ==============================
def draw_yolo_masks(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return None

    h, w = img.shape[:2]
    
    if not os.path.exists(label_path):
        print(f"No hay etiquetas para {image_path}")
        return img

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # Convertir coordenadas normalizadas a píxeles
        points = np.array([[int(coords[i] * w), int(coords[i + 1] * h)]
                           for i in range(0, len(coords), 2)], np.int32)
        points = points.reshape((-1, 1, 2))

        # Color por clase
        color = tuple(int(c) for c in colors[cls_id % len(colors)])

        # Dibujar contorno y relleno
        overlay = img.copy()
        cv2.fillPoly(overlay, [points], color=color)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
        cv2.putText(img, f"Class {cls_id}", tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


# ==============================
# PROCESAR TODAS LAS IMÁGENES
# ==============================
for file in os.listdir(images_dir):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(images_dir, file)
    label_path = os.path.join(labels_dir, file.rsplit(".", 1)[0] + ".txt")

    result_img = draw_yolo_masks(image_path, label_path)
    if result_img is None:
        continue

    if save_output:
        out_path = os.path.join(output_dir, file)
        cv2.imwrite(out_path, result_img)
        print(f"Guardado: {out_path}")
    else:
        cv2.imshow("YOLO Masks", result_img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break
    print(f"Creating mask for: {image_path}")
    break

cv2.destroyAllWindows()
