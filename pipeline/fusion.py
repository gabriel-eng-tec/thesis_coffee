import numpy as np
import cv2
import math

def gaussian_mask(center, shape, sigma):
    """Genera una máscara gaussiana centrada en una coordenada (x, y)."""
    x0, y0 = center
    h, w = shape
    y, x = np.ogrid[:h, :w]
    mask = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return mask

def compute_mask_from_yolo(boxes, image_shape, sigma_ratio=0.5):
    """
    Crea una máscara difusa M(x, y) a partir de las cajas detectadas por YOLO.
    boxes: lista de detecciones [x_center, y_center, width, height, conf]
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.float32)
    for (xc, yc, bw, bh, conf) in boxes:
        sigma = sigma_ratio * ((bw + bh) / 2)
        g = gaussian_mask((xc, yc), (h, w), sigma)
        mask = np.maximum(mask, conf * g)  # combina las máscaras
    return np.clip(mask, 0, 1)

def fuse_counts(yolo_boxes, density_map, image_shape):
    """
    Fusiona resultados de YOLO y CDMENet-Coffee.
    Retorna el conteo final fusionado y el residual estimado.
    """
    # Conteo YOLO
    N_yolo = len(yolo_boxes)

    # Crear máscara difusa M(x, y)
    M = compute_mask_from_yolo(yolo_boxes, image_shape)

    # Calcular mapa de densidad residual
    D_r = (1 - M) * density_map

    # Conteo residual
    N_r = np.sum(D_r)

    # Peso adaptativo alpha
    alpha = N_yolo / (N_yolo + N_r + 1e-6)

    # Conteo final
    N_final = alpha * N_yolo + (1 - alpha) * N_r

    return {
        "N_final": N_final,
        "N_yolo": N_yolo,
        "N_residual": N_r,
        "alpha": alpha
    }

# ==== Ejemplo de uso ====

# Simular detecciones YOLO (x_center, y_center, width, height, conf)
yolo_detections = [
    (150, 120, 40, 40, 0.95),
    (300, 200, 35, 35, 0.88),
    (400, 220, 30, 30, 0.90)
]

# Simular mapa de densidad (por ejemplo, salida normalizada de CDMENet)
H, W = 480, 640
density_map = np.random.rand(H, W) * 0.01  # valores pequeños para simular densidad
image_shape = (H, W)

# Fusión
results = fuse_counts(yolo_detections, density_map, image_shape)

print("Conteo YOLO:", results["N_yolo"])
print("Conteo residual:", results["N_residual"])
print("Alpha:", results["alpha"])
print("Conteo final fusionado:", results["N_final"])


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(density_map, cmap='jet')
plt.title("Mapa de densidad")

plt.subplot(1, 3, 2)
plt.imshow(fused, cmap='jet')
plt.title("Fusión YOLO + Densidad")

plt.subplot(1, 3, 3)
plt.imshow(fused > 0.5, cmap='gray')
plt.title("Regiones destacadas (umbral 0.5)")
plt.show()
