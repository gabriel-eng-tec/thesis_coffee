from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import random

# Cargar modelo preentrenado YOLOv8 segmentación (nano)
model = YOLO("yolov8n.pt")

# Abrir imagen
img_path = "../dataset/images/val/coffe_fruit001.jpg"
img = Image.open(img_path).convert("RGB")
img_rgba = img.convert("RGBA")

# Inferencia
results = model.predict(source=img_path, conf=0.25)

# Crear capa para máscaras
mask_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()      # [N, 4]
    scores = r.boxes.conf.cpu().numpy()     # [N]
    labels = r.boxes.cls.cpu().numpy()      # [N]
    
    # Máscaras si el modelo es de segmentación
    masks = r.masks.data.cpu().numpy() if r.masks is not None else None  # [N, H, W]

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < 0.25:
            continue
        
        # Dibujar máscara si existe
        if masks is not None:
            mask = masks[i]
            mask_img = Image.fromarray((mask * 255).astype("uint8")).convert("L")
            mask_resized = mask_img.resize(img.size)
            color = tuple(random.randint(100, 255) for _ in range(3)) + (100,)
            color_img = Image.new("RGBA", img.size, color)
            mask_layer = Image.composite(color_img, mask_layer, mask_resized)

# Combinar imagen original con máscaras
img_combined = Image.alpha_composite(img_rgba, mask_layer)

# Dibujar bounding boxes y texto
draw = ImageDraw.Draw(img_combined)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy()
    for box, score, label in zip(boxes, scores, labels):
        if score < 0.25:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"{int(label)} ({score:.2f})", fill="red", font=font)

# Guardar imagen final
output_path = "./mi_foto_yolo_masks.png"
img_combined.save(output_path)
print("Imagen guardada en:", output_path)
