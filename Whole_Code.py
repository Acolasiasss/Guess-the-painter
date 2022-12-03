!wget https://data.yanxishe.com/Art.zip
!unzip '/content/Art.zip'                                     #文件下载在 content 目录
!pip install efficientnet_pytorch

import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda:0")                          # tensor 赋值为GPU，计算时，唯一的GPU
print("************************** print(  torch.device('cuda:0') )  ******************")
print(device)

#数据处理
datadir = np.array(os.listdir('/content/train'))    #传入相应的路径，将会返回那个目录下的所有文件名。
#print(datadir)                                                   #返回数组类型
file = pd.read_csv('/content/train.csv')     #数据存储在 DateFrame

#函数 dictmap
def dictmap(data,num=1):
  labels = []
  fnames = []
  for i in data:
    name = int(i.split('.')[0])
    fnames.append(name)
    labels.append(file['label'][name])   # label 标签

  label_dict = dict(zip(fnames,labels))    #  { name : labels,  ...}
  # zip() 可以将两个可迭代对象中的对应元素打包成一个个元组，返回这些元组组成的列表
  #  dict 可以传入元组列表创建字典
  return label_dict

size = 300
#b3 1536
trans = {
        'train':
            transforms.Compose([              #一般用Compose把多个步骤整合到一起：
                transforms.Resize([350,350]),   #图片等比例缩放 h, w
                transforms.RandomCrop(300),   #660 600 93.125    随机剪裁，300期望输出的尺寸，可（100，100）
                transforms.RandomHorizontalFlip(),   #依概率 p(默认 0.5）水平翻转图片
                transforms.ToTensor(),           #PIL / numpy 格式的图像 转换为 tensor格式（张量，多维数组），并且归一化至[0-1]
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]), #标准化，(  -mean)/std  图片三个通道
                #transforms.RandomErasing(p=0.6,scale=(0.02,0.33),ratio=(0.3,0.33),value=0,inplace=False)
            ]),

        'val':
            transforms.Compose([
                transforms.Resize([300,300]),
                #transforms.RandomCrop(300),
                #transforms.RandomHorizontalFlip(),

                # transforms.RandomVerticalFlip(),  
                # transforms.RandomRotation(40),
                # transforms.RandomAffine(20),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        }
        
def default_loader(path):
    return Image.open(path).convert('RGB')   #用Image.open读出图像，用convert(‘RGB’)进行通道转换，RGB图片三通道

class dataset(Dataset):
    def __init__(self, data, lab_dict, loader=default_loader, mode='Train'):    #初始化，self 实例本身，_init_() 实例化时，自动执行
        super(dataset, self).__init__()   # 对继承自dataset的父类的属性初始化。用父类的初始化方法来初始化继承的属性。
        label = []

        for line in data:
            label.append(lab_dict[int(line.split('.')[0])])

        self.data = data
        self.loader = loader
        self.mode = mode
        self.transforms = trans[mode]
        self.label = label

    def __getitem__(self, index):    # 定义后，实例对象 P 就可以这样 P[key ]取值。
        fn = self.data[index]
        label = self.label[index]
        img = self.loader('/content/train/'+fn)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.from_numpy(np.array(label))

    def __len__(self):    #求类属性中 data 的长度
        return len(self.data)    
        
#从训练集中划分验证集val
def Train_Valloader(data):
    np.random.shuffle(data)
    lab_dict = dictmap(data)
    train_size = int((len(data)) * 0.8)#划分20%
    train_idx = data[:train_size]
    val_idx = data[train_size:]
    train_dataset = dataset(train_idx, lab_dict, loader=default_loader, mode='train')
    val_dataset = dataset(val_idx, lab_dict, loader=default_loader, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_size
    
model = EfficientNet.from_pretrained('efficientnet-b3')
print(model)
for name,param in model.named_parameters():
  if ("_conv_head" not in name) and ("_fc" not in name):
    param.requires_grad = False
    
num_trs = model._fc.in_features
model._fc = nn.Linear(num_trs,49)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# 学习率
lr = 0.01
# 随机梯度下降
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(pg,lr = lr)
#optimizer = torch.optim.SGD(model._fc.parameters(),lr=0.001,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)

#训练函数
def train(model,dataloader,size):
  model.train()
  running_loss = 0.0
  running_corrects = 0
  count = 0
  for inputs,classes in tqdm(dataloader):
    inputs = inputs.to(device)
    classes = classes.to(device)
    outputs = model(inputs)
    loss = criterion(outputs,classes)           
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _,preds = torch.max(outputs.data,1)
    # statistics
    running_loss += loss.data.item()
    running_corrects += torch.sum(preds == classes.data)
    count += len(inputs)
    #print('Training: No. ', count, ' process ... total: ', size)
  scheduler.step()
  epoch_loss = running_loss / size
  epoch_acc = running_corrects.data.item() / size
  
  # 测试函数
def val(model,dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    best_acc = 60
    save_path = '/content/art_effB3.pth'
    with torch.no_grad():  # 不需要梯度，减少计算量
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100. * correct / total
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)
        print('accuracy on test set: %d %% ' % accuracy)
        return accuracy
        
accuracy_list = []
epoch_list = []
acc=0
train_loader, val_loader,train_size = Train_Valloader(datadir)
for epoch in range(50):
  train(model,train_loader,train_size)
  acc = val(model,val_loader)
  accuracy_list.append(acc)
  epoch_list.append(epoch)
        
plt.plot(epoch_list, accuracy_list)
plt.xlabel(epoch)
plt.ylabel(accuracy_list)
plt.show()

#test
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = []
        self.name = []
        self.test_transform = transforms.Compose([
                transforms.Resize([300,300]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        filepath = '/content/test'
        for filename in tqdm(os.listdir(filepath)):
            image = Image.open(os.path.join(filepath, filename)).convert('RGB')
            image = self.test_transform(image)
            self.images.append(image)
            self.name.append(filename)

    def __getitem__(self, item):
        return self.images[item]

    def __len__(self):
        images = np.array(self.images)
        len = images.shape[0]
        return len

test_datasets = Dataset()
device = torch.device("cuda:0")
testloaders = torch.utils.data.DataLoader(test_datasets, shuffle=False)
testset_sizes = len(test_datasets)
#加载之前保存的pth
model = EfficientNet.from_name('efficientnet-b3')
f = model._fc.in_features
model._fc = nn.Linear(f,49)
model_weight_path = "/content/art_effB3.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.to(device)


dic = {}


def test(model):
    model.eval()
    cnt = 0
    for inputs in tqdm(testloaders):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        key = test_datasets.name[cnt].split('.')[0]
        dic[key] = preds[0]
        cnt += 1
        with open("/content/result.csv", 'a+') as f:
            f.write("{},{}\n".format(key, dic[key]))
test(model)