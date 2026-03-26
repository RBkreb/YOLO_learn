import warnings
from ultralytics import YOLO
warnings.filterwarnings("ignore")

if __name__=="__main__":
    model=YOLO(model=r"ultralytics\ultralytics\cfg\models\12\yolo12.yaml")
    model.load(r"model\yolo12n.pt")
    model.train(data=r"datasets\DesktopSensor\data.yaml",
                 imgsz=640,
                epochs=100,
                batch=16,
                scale=0.5,
                workers=0,
                device=0,
                mosaic=1.0,
                mixup=0,
                close_mosaic=20,
                project="runs/desktopsensor")