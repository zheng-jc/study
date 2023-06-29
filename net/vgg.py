import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

# 用一个字典存储
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

# 两层3x3的卷积核感受野为5x5，三层3x3的卷积核感受野为7x7呢
class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, class_num)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_features(cfg:list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'vgg11': [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13': [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16': [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19': [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
}

def vgg(model_name="vgg16", **kwargs): # **kwargs 可变长度字典当中
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model
vgg13 = vgg(model_name='vgg13',class_num=10, init_weights=True).to(device)
print(vgg13)


loss_function=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(vgg13.parameters(),lr=0.001)

epochs=5

best_acc=0.0
for epoch in range(epochs):
    vgg13.train()
    running_loss=0
    for step,data in enumerate(trainloader,start=0):
        images,labels=data
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        output=vgg13(images)
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
    vgg13.eval()
    acc=0.0
    with torch.no_grad():
        for data in testloader:
            images,labels=data
            images,labels=images.to(device),labels.to(device)
            output=vgg13(images)
            loss=loss_function(output,labels)
            predict_y=torch.max(output,dim=1)[1]
            acc+=(predict_y==labels).sum().item()
        accuracy_test=acc/val_num
        if accuracy_test>best_acc:
            best_acc=accuracy_test
        print("[{}/{},test_accuracy:{}  best_acc;{}]".format(epoch+1,epochs,accuracy_test,best_acc))
print('结束训练')
