import pandas
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms

if __name__ == '__main__':
    classname = pandas.read_excel('class_names.xls')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(pretrained=False).to(device)
    checkpoint = torch.load('model/resmodel.pt')
    model.load_state_dict(checkpoint['model'])
    model.eval() 
    img = Image.open('data/tmaegg.webp').convert('RGB')
    img = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52011104, 0.44459117, 0.30962785], std=[0.25595631, 0.25862494, 0.26925405])
    ])(img)
    img = img.reshape(1, 3, 128, 128).to(device)
    pre = model(img)
    _, predicted = torch.max(pre, 1)
    correct = torch.argmax(pre.data, 1)
    i = correct.item()
    print("{}:{:.2f}%".format(i, pre[0][i]*10))
    print(classname["chname"][correct.item()])
