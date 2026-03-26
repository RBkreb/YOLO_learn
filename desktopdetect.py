import warnings
from ultralytics import YOLO
warnings.filterwarnings("ignore")
if __name__=="__main__":
    model=YOLO(r"runs\desktopsensor\train15\weights\best.pt")
    model.predict(r"data\PP24.png",save=True,save_txt=True,save_conf=True)