import torch
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image, ImageDraw
import sys
from pathlib import Path

# ================================
# 1. Cargar DINOv3 backbone
# ================================
repo_root = Path(__file__).parent / "../models/dinov3"
sys.path.append(str(repo_root))

# Usar torch.hub desde repo local
dinov3 = torch.hub.load(
    repo_or_dir=str(repo_root),
    model="dinov3_vits16",  # también hay dinov3_vitb16, vitl16...
    source="local",
    pretrained=False,
    backbone_weights="dinov3_vitb16.pth"
)

# Añadir atributo out_channels para Faster R-CNN
dinov3.out_channels = 384  # ViT-S/16 → 384, ViT-B/16 → 768

# ================================
# 2. Definir modelo Faster R-CNN
# ================================
rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

model = FasterRCNN(
    backbone=dinov3,
    num_classes=2,  # 1 clase (fruto) + background
    rpn_anchor_generator=rpn_anchor_generator
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# ================================
# 3. Transformar imagen
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # DINOv3 requiere 224x224
    transforms.ToTensor(),
])

img_path = "../dataset/images/val/coffe_fruit001.jpg"
img = Image.open(img_path).convert("RGB")
inp = transform(img).unsqueeze(0).to(device)

# ================================
# 4. Inferencia (sin entrenar)
# ================================
with torch.no_grad():
    preds = model(inp)

print("Predicciones brutas:", preds)

# ================================
# 5. Dibujar cajas (aunque sean aleatorias sin entrenar)
# ================================
draw = ImageDraw.Draw(img)
for box, score, label in zip(preds[0]["boxes"], preds[0]["scores"], preds[0]["labels"]):
    if score > 0.5:  # threshold de confianza
        x1, y1, x2, y2 = box.cpu().numpy()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{label.item()} ({score:.2f})", fill="red")

img.save("./data_out/dinov3_fasterrcnn_out.jpg")
print("Imagen guardada en dinov3_fasterrcnn_out.jpg")
