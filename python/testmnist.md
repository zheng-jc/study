初代网络

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,10)
        )

    def forward(self, x):
        x=self.model(x)
        return x

```



使用CNN

```python
# 定义LeNet网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model=nn.Sequential(
            # MNIST数据集大小为28x28，要先做padding=2的填充才满足32x32的输入大小
            nn.Conv2d(1,6,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self, x):
        x=self.model(x)
        return x
```





完整代码

```python
import torch
import torchvision

import torch.nn as nn
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader

# 先定义一个绘图工具
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)),data,color = 'blue')
    plt.legend(['value'],loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

device=torch.device('cuda' if torch.cuda.is_available() else "cpu")

# 定义LeNet网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model=nn.Sequential(
            # MNIST数据集大小为28x28，要先做padding=2的填充才满足32x32的输入大小
            nn.Conv2d(1,6,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self, x):
        x=self.model(x)
        return x

epoch=20
batch_size=64
lr=0.001


transforms = torchvision.transforms.Compose([
                                # torchvision.transforms.Normalize((0.1307,),(0.3081,)
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
                                torchvision.transforms.ToTensor()])
# 导入数据集
traindata=torchvision.datasets.MNIST(root='./dataset', train=True, transform=transforms,download=True)
testdata=torchvision.datasets.MNIST(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),download=True)

test_size=len(testdata)

# 加载数据集
trainloader=DataLoader(traindata,batch_size=batch_size,shuffle=True)
testloader=DataLoader(testdata,batch_size=batch_size,shuffle=False)

net=LeNet().to(device)

loss_fn=nn.CrossEntropyLoss().to(device)

optimizer=torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9)

train_loss=[]
precision=[]
train_step=0
for epoch in range(epoch):
    net.train()
    sum_loss=0
    for data in trainloader:
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)

        # 更新梯度
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()

        train_step+=1
        sum_loss+=loss.item()
        if train_step % 100==99:
            print("[epoch:{},轮次：{}，sum_loss:{}]".format(epoch+1,train_step,sum_loss/100))
            train_loss.append(sum_loss/100)
            sum_loss=0

    net.eval()
    with torch.no_grad():
        correct=0
        # total=0
        accuracy=0
        for data in testloader:
            images, labels=data
            images,labels=images.to(device),labels.to(device)
            outputs=net(images)
            # _,predicted=torch.max(outputs.data,1)
            # total+=labels.size(0)
            # correct+=(predicted==labels).sum()
            correct+=(outputs.argmax(1)==labels).sum()
        accuracy=correct/test_size
        print("第{}个epoch的识别准确率为：{}".format(epoch+1,accuracy))
        precision.append(accuracy.cpu())

# plot_curve(train_loss)
# plot_curve(precision)
torch.save(net,'net_lenet_20.pth')
```



20个epoch

![1682480216784](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1682480216784.png)

40个epoch

![1682484942496](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1682484942496.png)





验证阶段

```python
import numpy as np
import torch
from torch import nn

import torchvision
from PIL import Image

import math

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.model=nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28*28,256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256,64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64,10)
#         )
#
#     def forward(self, x):
#         x=self.model(x)
#         return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model=nn.Sequential(
            # MNIST数据集大小为28x28，要先做padding=2的填充才满足32x32的输入大小
            nn.Conv2d(1,6,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self, x):
        x=self.model(x)
        return x

model1=torch.load('net_lenet_40.pth',map_location=torch.device('cpu'))
model2=torch.load('net_lenet_40_1.pth',map_location=torch.device('cpu'))
# model=torch.load('net_8.pth')

#
# image_path='./image_mnist/1.png'
# image=Image.open(image_path)
# image=image.convert('L')


transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((28,28)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,),(0.3081,))

])

# image=transform(image)
#
#
# image=torch.reshape(image,(1,1,28,28))
#
# model.eval()
# with torch.no_grad():
#     output=model(image)
# print(output)
# print("预测结果为：{}".format(output.argmax(1).item()))






def isPrime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# if isPrime(output.argmax(1).item()):
#     print("{}是质数".format(output.argmax(1).item()))
# else:
#     print("{}不是质数".format(output.argmax(1).item()))




# 10张图
for i in range(10):
    image_path = './image_mnist/{}.png'.format(i)
    image = Image.open(image_path)
    image = image.convert('L')

    image=transform(image)
    image = torch.reshape(image, (1, 1, 28, 28))
    model1.eval()
    with torch.no_grad():
        output1 = model1(image)
    model2.eval()
    with torch.no_grad():
        output2 = model2(image)
    # print(output)
    output=output1+output2
    if i==output.argmax(1).item():
        print("真实标签为{}，预测结果为：{}，预测正确".format(i, output.argmax(1).item()))
        if isPrime(output.argmax(1).item()):
            print("{}是质数".format(output.argmax(1).item()))
        else:
            print("{}不是质数".format(output.argmax(1).item()))
    else:
        print("真实标签为{}，预测结果为：{}，预测错误".format(i, output.argmax(1).item()))

```





