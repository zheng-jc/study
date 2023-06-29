import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

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

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(AlexNet, self).__init__()
        self.features=nn.Sequential(
            # 如果是更精细化的padding操作，使用nn.ZeroPad2d((1,2,1,2)),表示左侧补一列，右侧补2列，上方补1行，下方补2行
            # nn.Conv2d(3,48,kernel_size=11,stride=4,padding=(1,2)),    (1,2)表示上下方，各补一行0，左右各补2列0
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2),  # input(3,224,224)  output(48,55,55)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),               # output(48,27,27)
            nn.Conv2d(48,128,kernel_size=5,padding=2),          # output(128,27,27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),               # output(128,13,13)
            nn.Conv2d(128,192,kernel_size=3,padding=1),         # output(192,13,13)
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3,padding=1),         # output(192,13,13)
            nn.ReLU(inplace=True),
            nn.Conv2d(192,128,kernel_size=3,padding=1),         # output(128,13,13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),               # output(128,6,6)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)
        x=self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# z=torch.randn(10,3,224,224)
alexnet=AlexNet().to(device)

loss_function=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(alexnet.parameters(),lr=0.001)

epochs=5

best_acc=0.0
for epoch in range(epochs):
    alexnet.train()
    running_loss=0
    for step,data in enumerate(trainloader,start=0):
        images,labels=data
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        output=alexnet(images)
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
    alexnet.eval()
    acc=0.0
    with torch.no_grad():
        for data in testloader:
            images,labels=data
            images,labels=images.to(device),labels.to(device)
            output=alexnet(images)
            loss=loss_function(output,labels)
            predict_y=torch.max(output,dim=1)[1]
            acc+=(predict_y==labels).sum().item()
        accuracy_test=acc/val_num
        if accuracy_test>best_acc:
            best_acc=accuracy_test
        print("[{}/{},test_accuracy:{}  best_acc;{}]".format(epoch+1,epochs,accuracy_test,best_acc))
print('结束训练')
