import torch
import torchvision
checkpoint=torch.load('model/resmodel.pt')
model = torchvision.models.resnet50(pretrained=False)
model.load_state_dict(checkpoint['model'])
model.eval() 
input_names = ['input']
output_names = ['output']
 
x = torch.randn(1,3,128,128)
 
torch.onnx.export(model, x, 'best.onnx', input_names=input_names, output_names=output_names)