import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

traindata=torchvision.datasets.CIFAR10('../dataset',transform=transform,train=True,download=True)
testdata=torchvision.datasets.CIFAR10('../dataset',transform=transform,train=False,download=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainloader = DataLoader(traindata,batch_size=36,shuffle=True,num_workers=0)
testloader = DataLoader(testdata,batch_size=10000,shuffle=False,num_workers=0)

testdata_iter=iter(testloader)
test_image, test_label=next(testdata_iter)
test_image,test_label=test_image.to(device),test_label.to(device)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
print(len(testdata))

# def imshow(img):
#     img=img/2+0.5
#     nping=img.numpy()
#     plt.imshow(np.transpose(1,2,0))
#     plt.show()
#
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
#
# imshow(torchvision.utils.make_grid(test_image))

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1=nn.Conv2d(3,16,5)
        self.pool1=nn.MaxPool2d(2,2)    # 第一个参数是池化核的大小，第二个参数是步长，步长默认是池化核的大小
        self.conv2=nn.Conv2d(16,32,5)
        self.pool2=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(32*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

        # self.model=nn.Sequential(
        #     nn.Conv2d(3,16,5),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2,2),
        #     nn.Conv2d(16,32,5),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2,2),
        #     nn.Flatten(),
        #     nn.Linear(32*5*5,120),
        #     nn.Linear(120,84),
        #     nn.Linear(84,10)
        # )

    def forward(self, x):
        x=F.relu(self.conv1(x))     # input(3,32,32)    output(16,28,28)
        x=self.pool1(x)             # output(16,14,14)
        x=F.relu(self.conv2(x))     # output(32,10,10)
        x=self.pool2(x)             # output(32,5,5)
        x=x.view(-1,32*5*5)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

        # x=self.model(x)
        # return x




# z=torch.randn([10,3,32,32])

lenet=Lenet().to(device)
loss_function=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(lenet.parameters(),lr=0.001)

epochs=5
for epoch in range(epochs):
    lenet.train()
    running_loss=0
    # enumerate函数会返回每一批的数据，还会返回这一批数据所对应的步数
    for step,data in enumerate(trainloader,start=0):
        images,labels=data
        images,labels=images.to(device),labels.to(device)
        output=lenet(images)
        loss=loss_function(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if step%500 == 499:
            with torch.no_grad():
                output = lenet(test_image)
                predict_y=torch.max(output,dim=1)[1]
                accuracy = ((predict_y==test_label).sum().item()/test_label.size(0))
                print("[%d:%5d]:train_loss:%.3f    test_accuracy:%.3f" %(epoch+1,step+1,running_loss/500,accuracy))
                running_loss=0.0

# 保存模型

# 加载模型


# 预测

transforms=torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
im=Image.open("../img/plane.jpg")
im=transforms(im) # [C,H,W]
im=torch.unsqueeze(im,dim=0) # [N,C,H,W]
im=im.to(device)

with torch.no_grad():
    output = lenet(im)
    predict=torch.max(output, dim=1)[1].data.cpu()
    # predict=torch.softmax(output, dim=1)
print(classes[int(predict)])