import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Dataset personalizado para imágenes de café
class coffeeDensityDataset(Dataset):
    def __init__(self, image_dir, density_maps_dir, yolo_model=None, transform=None):
        self.image_dir = image_dir
        self.density_maps_dir = density_maps_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform
        self.yolo_model = yolo_model  # Modelo YOLO ya entrenado

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crear mapa YOLO (mask binaria de detecciones)
        if self.yolo_model:
            results = self.yolo_model.predict(source=img_path, conf=0.5, verbose=False)
            yolo_mask = np.zeros(image.shape[:2], dtype=np.float32)
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)
                yolo_mask[y1:y2, x1:x2] = 1.0
            yolo_mask = np.expand_dims(yolo_mask, axis=2)
        else:
            yolo_mask = np.zeros(image.shape[:2] + (1,), dtype=np.float32)

        # Concatenar imagen + YOLO mask como input
        input_image = np.concatenate([image.astype(np.float32)/255.0, yolo_mask], axis=2)  # [H, W, 4]

        if self.transform:
            input_image = self.transform(input_image)

        # Cargar mapa de densidad real si existe
        density_path = os.path.join(self.density_maps_dir, self.image_files[idx].replace('.jpg', '.npy'))
        density_map = np.load(density_path) if os.path.exists(density_path) else np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        density_map = torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)

        return input_image, density_map
