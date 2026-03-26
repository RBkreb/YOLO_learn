import warnings
from ultralytics import YOLO
warnings.filterwarnings("ignore")
if __name__=="__main__":
    model=YOLO(r"runs\calorie\train\weights\best.pt")
    model.predict(r"data\biryani1.webp",save=True,save_txt=True,save_conf=True)