import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = {
    "train": torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224),
                                             torchvision.transforms.RandomHorizontalFlip(),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                            ]),
    "val": torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])}


train_data=torchvision.datasets.CIFAR10(root='./dataset',transform=data_transform["train"],train=True,download=True)
test_data=torchvision.datasets.CIFAR10(root='./dataset',transform=data_transform["val"],train=False,download=True)

trainloader=DataLoader(train_data,batch_size=64)
testloader=DataLoader(test_data,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()