from ultralytics import YOLO

# ============================================================
# 🧩 MODELO YOLO
# ============================================================
class YOLO():
    def __init__(self, model_name="yolov8m.pt", pretrained=True):
        self.model = YOLO(f"{model_name}" if pretrained else f"{model_name}_n.pt")
