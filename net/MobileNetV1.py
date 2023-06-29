
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

class DepthSeperableConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super(DepthSeperableConv2d, self).__init__()
        self.depthwise=nn.Sequential(
            nn.Conv2d(input_channels,input_channels,3,stride,1,groups=input_channels,bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,1,bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        x=self.depthwise(x)
        x=self.pointwise(x)
        return x

class MobileNetV1(nn.Module):

    def __init__(self, width_multiplier=1, class_num=1000):
        super(MobileNetV1, self).__init__()
        alpha = width_multiplier
        self.conv1=nn.Sequential(
            nn.Conv2d(3,int(alpha*32), 3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(alpha*32)),
            nn.ReLU6(inplace=True)
        )
        self.features=nn.Sequential(
            DepthSeperableConv2d(int(alpha*32),int(alpha*64),1),
            DepthSeperableConv2d(int(alpha*64),int(alpha*128),2),
            DepthSeperableConv2d(int(alpha*128),int(alpha*128),1),
            DepthSeperableConv2d(int(alpha*128),int(alpha*256),2),
            DepthSeperableConv2d(int(alpha*256),int(alpha*256),1),
            DepthSeperableConv2d(int(alpha*256),int(alpha*512),2),
            DepthSeperableConv2d(int(alpha*512),int(alpha*512),1),
            DepthSeperableConv2d(int(alpha*512),int(alpha*512),1),
            DepthSeperableConv2d(int(alpha*512),int(alpha*512),1),
            DepthSeperableConv2d(int(alpha*512),int(alpha*512),1),
            DepthSeperableConv2d(int(alpha*512),int(alpha*512),1),
            DepthSeperableConv2d(int(alpha*512),int(alpha*1024),2),
            DepthSeperableConv2d(int(alpha*1024),int(alpha*1024),2),
        )
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(int(alpha*1024),class_num)
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# net=MobileNetV1()
# print(net)
# z=torch.randn(1,3,224,224)
# output=net(z)
# print(output)


data_transform = {
    "train": torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224),
                                             torchvision.transforms.RandomHorizontalFlip(),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
    "val": torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}


traindata=torchvision.datasets.CIFAR10(root='../dataset',transform=data_transform["train"],download=True,train=True)
testdata=torchvision.datasets.CIFAR10(root='../dataset',transform=data_transform["val"],download=True,train=False)

val_num=len(testdata)

batch_size=128
trainloader=DataLoader(traindata,batch_size=batch_size,shuffle=True,num_workers=0)
testloader=DataLoader(testdata,batch_size=batch_size,shuffle=False,num_workers=0)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net=MobileNetV1().to(device)

loss_function=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(net.parameters(),lr=0.001)

epochs=5

best_acc=0.0
for epoch in range(epochs):
    net.train()
    running_loss=0
    for step,data in enumerate(trainloader,start=0):
        images,labels=data
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        output=net(images)
        loss=loss_function(output,labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss +=loss.item()
        rate=(step+1)/len(trainloader)
        a='*' * int(rate*50)
        b='*' * int((1-rate)*50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100),a,b,loss),end='')
    print()

    # 因为模型用了dropout方法，所以要写这个方法，测试的时候，模型会屏蔽掉dropout
    net.eval()
    acc=0.0
    with torch.no_grad():
        for data in testloader:
            images,labels=data
            images,labels=images.to(device),labels.to(device)
            output=net(images)
            loss=loss_function(output,labels)
            predict_y=torch.max(output,dim=1)[1]
            acc+=(predict_y==labels).sum().item()
        accuracy_test=acc/val_num
        if accuracy_test>best_acc:
            best_acc=accuracy_test
        print("[{}/{},test_accuracy:{}  best_acc;{}]".format(epoch+1,epochs,accuracy_test,best_acc))
print('结束训练')
