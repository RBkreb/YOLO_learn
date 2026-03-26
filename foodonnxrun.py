import pandas as pd
import numpy as np
from PIL import Image
import onnxruntime as ort
from torchvision.transforms import functional as F

onnx_path = 'model/best.onnx'      
excel_path = 'class_names.xls'
img_path = 'data/tmaegg.webp'
input_size = 128
crop_size = 500
mean = [0.52011104, 0.44459117, 0.30962785]
std  = [0.25595631, 0.25862494, 0.26925405]

classname = pd.read_excel(excel_path)
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(onnx_path, providers=providers)
input_name = session.get_inputs()[0].name         
output_name = session.get_outputs()[0].name

img = Image.open(img_path).convert('RGB')
w, h = img.size
left = (w - crop_size) // 2
top  = (h - crop_size) // 2
img = img.crop((left, top, left + crop_size, top + crop_size))
img = F.resize(img, [input_size, input_size])
img = F.to_tensor(img)
img = F.normalize(img, mean=mean, std=std)
img = img.unsqueeze(0).numpy()

logits = session.run([output_name], {input_name: img})[0] 
prob = logits[0]
pred_id = int(np.argmax(prob))
confidence = prob[pred_id] * 10         
ch_name = classname["chname"][pred_id]
print("{}:{:.2f}%".format(pred_id, confidence))
print(ch_name)