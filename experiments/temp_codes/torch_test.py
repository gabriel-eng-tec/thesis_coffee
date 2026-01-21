import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
import random

# Cargar modelo preentrenado (COCO)
model = maskrcnn_resnet50_fpn(weights="DEFAULT").eval().to("cuda")

# Abrir imagen
img_path = "../dataset/images/val/coffe_fruit001.jpg"
img = Image.open(img_path).convert("RGB")
img_rgba = img.convert("RGBA")

# Convertir a tensor
tensor = F.to_tensor(img).unsqueeze(0).to("cuda")

# Inferencia
with torch.no_grad():
    outputs = model(tensor)

# Extraer datos
boxes = outputs[0]["boxes"].cpu()
labels = outputs[0]["labels"].cpu()
scores = outputs[0]["scores"].cpu()
masks = outputs[0]["masks"].cpu()  # [N, 1, H, W]

# Crear capa para máscaras
mask_layer = Image.new("RGBA", img.size, (0,0,0,0))

for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
    if score > 0.5:
        # Máscara
        mask_img = Image.fromarray((mask[0].numpy() * 255).astype("uint8")).convert("L")
        mask_resized = mask_img.resize(img.size)
        
        # Color aleatorio semi-transparente
        color = tuple(random.randint(100, 255) for _ in range(3)) + (100,)
        color_img = Image.new("RGBA", img.size, color)
        mask_layer = Image.composite(color_img, mask_layer, mask_resized)

# Combinar la capa de máscaras con la imagen original
img_combined = Image.alpha_composite(img_rgba, mask_layer)

# Dibujar bounding boxes y texto sobre la imagen combinada
draw = ImageDraw.Draw(img_combined)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
    #if score > 0.5:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    draw.text((x1, y1 - 10), f"{label.item()} ({score:.2f})", fill="red", font=font)

# Guardar imagen final
output_path = "./mi_foto_maskrcnn_masks.png"
img_combined.save(output_path)
print("Imagen guardada en:", output_path)
