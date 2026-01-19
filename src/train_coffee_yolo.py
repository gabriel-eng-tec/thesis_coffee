import torch
#
from ultralytics import YOLO

print("GPU disponible:", torch.cuda.is_available())
print("Dispositivo actual:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Limpiar cache
torch.cuda.empty_cache()

image_dir_train = "../dataset/coffee_Fruit_Maturity_yolo"

model_dir = "../models/yolov8m"
project_name = f"{model_dir}/runs/segment"
experiment_name = "coffee_segmentation_v8m"

model = YOLO(f"{model_dir}/yolov8m-seg.pt")
model.train(
    data=f"{image_dir_train}/data.yaml",
    epochs=50,
    imgsz=1024, #640?
    batch=8, # Aguanta 12
    name=experiment_name,
    project=project_name,
    pretrained=True,
    augment=True
)

# Cargar el mejor modelo
best_model_path = f"{project_name}/{experiment_name}/weights/best.pt"
model = YOLO(best_model_path)

#####
# Predecir sobre imágenes nuevas
#####
image_dir_test = "../dataset/icafe/images/val"

results = model.predict(
    source=image_dir_test,
    save=True,
    show=False,    # pon True si estás en entorno gráfico
    conf=0.5,
    project="../outputs",
    name="coffee_seg_test",
    exist_ok=True  # Reutiliza la carpeta si ya existe
)

# Mostrar los paths de salida
for r in results:
    print(r.path)
