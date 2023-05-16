#### 0.pytorch加载数据集

- Dataset和Dataloader的区别

![1677055178945](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677055178945.png)

##### 0.1加载cifar10数据集

```python
import torchvision

traindata=torchvision.datasets.CIFAR10('./dataset',train=True,download=True)
testdata=torchvision.datasets.CIFAR10('./dataset',train=False,download=True)

# 这里输出测试集有哪些类别，可以debug看到这些属性
print(testdata.classes)
# 每个数据都由一张32x32的图片，和它正确对应的标签组成
print(testdata[0])

img,target=testdata[0]
print(target)
# 这里可以直接打印该图片的效果
img.show()
```

![1681892238770](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681892238770.png)







#### 1.tensorboard的使用

先要安装tensorboard库

代码演示：

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# tensorboard的使用，主要是要用于一些各个阶段的数据更直观的显示

# 创建实例对象
writer = SummaryWriter("logs") # 这里写入文件的存放路径
# 将文件存在logs这个文件夹下面
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)
    #参数分别对应，标题，y轴，x轴
writer.close()

# 使用tensorboard --logdir=logs打开logs文件里的图像显示
# 先cd切换到logs的根目录下

# 使用tensorboard --logdir=logs --port=6007可以指定端口，避免端口冲突
```

打开terminal终端运行窗口，输入如下命令

![1677576749349](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677576749349.png)

运行结果：

![1677576663015](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677576663015.png)



#### 2.transform的使用

![1677153601115](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677153601115.png)

- 关注输入和输出的类型
- 可以print看数据类型

#### 3.dataloader的使用

代码演示：

```python
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入数据，这里要先将PIL的数据转化成tensor类型
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 使用dataloader，指定数据集的名字，一次抓取的数量，是否打乱，设置进程数，是否将最后一批不满足batch_size数量的data保留
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


writer = SummaryWriter("logs") # 这里添加，生成的文件的存放到位置

# 随机抓取，shuffle设置为true
for epoch in range(2):
    step = 0
    for data in test_loader:
        img, target = data
        # 这里使用的是add_images,因为这里一次读取的图片不止一张
        #writer.add_images("test_loader_64", img, step)
        # 这里对标签进行一个替换
        # 这里为防止出现刷新是tensorboard的显示不一致，可以将logs目录下的文件都删除，再运行代码
        writer.add_images("epoch：{}".format(epoch), img, step)
        step = step + 1
writer.close()
```



运行结果：

![1677402095080](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677402095080.png)



#### 4.卷积层

![1677402193610](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677402193610.png)

![1677402248081](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677402248081.png)

代码演示：

```python
import torch
import torch.nn.functional as nn
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

# 卷积层的输入有限制，首先是minibatch，通道数，高，宽
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

output = nn.conv2d(input,kernel,stride=1)
print(output)

output = nn.conv2d(input,kernel,stride=2)
print(output)

output = nn.conv2d(input,kernel,stride=1,padding=1)
print(output)
```

运行结果：

![1677402391912](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677402391912.png)



- 卷积对图像(CIFAR10)的处理

代码演示：

```python

import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64) # 使用dataloader抓取数据



class zjctest(nn.Module):
    def __init__(self):
        super(zjctest, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)


    def forward(self, x):
        x = self.conv1(x)
        return x

zjcdata = zjctest()

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, target = data
    output = zjcdata(imgs)
    print(imgs.shape)

    # 如果要让输入图像和输出图像的大小保持一致，需要进行padding填充
    print(output.shape)
    writer.add_images("input",imgs,step,)

# torch.Size([64, 6, 30, 30])-》([xxx,3,30,30]),有点类似于把立体的每一层放在一个平面里
    output = torch.reshape(output,(-1,3,30,30))
    # 这里数据通道数为6，彩色图像数3个通道数，需变化shape
    writer.add_images("output",output,step)
    step = step + 1

writer.close()
```



运行结果：

![1677402744201](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677402744201.png)



#### 5.池化层

![1677390499282](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677390499282.png)

- 池化操作的步长为池化核的大小
- 当池化操作不满足池化核大小是，有两种模式，区别如上图
- ceil_model默认情况是false



代码演示

```python
import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)# max_pool2d" not implemented for 'Long,这里要设置数据格式

input = torch.reshape(input, (-1, 1, 5, 5)) # 这里对输入有要求

print(input.shape)
class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool1(input)
        return output
zjc = zjc()
output = zjc(input)
print(output)

```

输出结果：

![1677392981321](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677392981321.png)



池化层的直观感受：提取主要特征

> 最大池化的直观感受，有点类似于输入是1080p，输出是720p，720p也能满足我们的绝大多数需求，在满足看到视频内容的同时，文件尺寸会大大的缩小，加快我们的训练

代码实现：

```python
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 最大池化的直观感受，有点类似于输入是1080p，输出是720p，720p也能满足我们的绝大多数需求，在满足看到视频内容的同时，文件尺寸会大大的缩小
dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)  # 不知道还需要什么参数时，按ctrl+p 有提示




# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)# max_pool2d" not implemented for 'Long,这里要设置数据格式
#
# input = torch.reshape(input, (-1, 1, 5, 5)) # 这里对输入有要求
#
# print(input.shape)
class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool1(input)
        return output
zjc = zjc()
# output = zjc(input)
# print(output)

writer = SummaryWriter("../maxpool_logs") # 这里写日志存放的位置，最后一个是存放的文件夹的名字
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = zjc(imgs) # 池化操作，池化不会像卷积一样改变通道数
    writer.add_images("output", output, step)
    step = step + 1

writer.close()


```

对cifar10数据进行池化操作运行效果：

![1677394656217](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677394656217.png)

#### 6.非线性激活

ReLU

![1677397676786](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677397676786.png)

ReLU参数中inplace，默认为false，保证原始数据的不丢失

![1677397748642](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677397748642.png)

代码实现：

```python
import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))

print(input.shape)

class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input): # 这里不要忘记了参数传入
        output = self.relu1(input)
        return output

zjc = zjc()
output = zjc(input)
print(output)
```

输出结果：

![1677397872536](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677397872536.png)

非线性变换对图像(CIFAR10)的处理sigmoid

代码演示：

```python
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset",train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, 64)
class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        #self.relu1 = ReLU()
        self.simoid1 = Sigmoid()

    def forward(self, input): # 这里不要忘记了参数传入
        #output = self.relu1(input)
        output = self.simoid1(input)
        return output

zjc = zjc()

writer = SummaryWriter("../ReLU_logs")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = zjc(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
```

运行结果：

![1677398893457](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677398893457.png)

#### 7.神经网络的其他层

线性层

代码演示

```python
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        self.linear1 = Linear(in_features=196608, out_features=10)

    def forward(self, input):
        output = self.linear1(input)
        return output


zjc = zjc()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))# 展成一维的,这里可以用torch.flatten替代
    output = torch.flatten(imgs)

    print(output.shape)
    output = zjc(output)
    print(output.shape)
```

输出结果：

![1677401708216](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677401708216.png)

#### 8.完整的神经网络模型

##### 8.0模型搭建的方式

- 使用nn.Sequential( )

```python
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear=nn.Linear(32*7*7,10)


    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x
```



- 先创建容器类，然后使用add_module函数向里面添加新模块

```python
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Sequential()
        self.conv1.add_module("conv1",nn.Conv2d(1,16,5,1,2))
        self.conv1.add_module("relu1",nn.ReLU())
        self.conv1.add_module('maxpool1',nn.MaxPool2d(2))


        self.conv2=nn.Sequential()
        self.conv2.add_module('conv2',nn.Conv2d(16,32,5,1,2))
        self.conv2.add_module('relu2',nn.ReLU())
        self.conv2.add_module('maxpool2',nn.MaxPool2d(2))
        
        self.linear=nn.Linear(32*7*7,10)


    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x
```



- 利用nn.Function中的函数

```python
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(1,16,5,1,2)

        self.conv2=nn.Conv2d(16,32,5,1,2)

        self.fltten=nn.Flatten()
        self.linear=nn.Linear(32*7*7,10)


    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x),inplace=True),2)
        x=F.max_pool2d(F.relu(self.conv2(x),inplace=True),2)
        # x=x.view(x.size(0),-1)
        x=self.fltten(x)
        x=self.linear(x)
        return x
```







![1677407757242](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677407757242.png)

代码演示：

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5,padding=2)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(32, 32, 5, padding=2) # 这里的padding值要指定传参
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

zjc = zjc()
print(zjc)
```

运行结果：

![1677408710392](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677408710392.png)



##### 8.1检验模型是否正确

- 如何检查模型是否正确

  我们可以指定我们输入数据的一个形状
  
  输入一个指定tensor的大小，可以检验模型能否正确跑通
  
  

比如指定x=torch.ones(64, 3, 32, 32)

这里的64是batch_size的大小，3是指输入的通道数，对应第一个卷积层的输入通道数，32*32是指输入图片的大小





代码演示：

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5,padding=2)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

zjc = zjc()
print(zjc)

input = torch.ones((64, 3, 32, 32))
output = zjc(input)
print(output.shape)
```

运行结果：

![1677409171294](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677409171294.png)



- sequential的使用：使代码更简洁

代码演示：

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

zjc = zjc()
print(zjc)

input = torch.ones((64, 3, 32, 32))
output = zjc(input)
print(output.shape)
```

运行结果：

![1677409562775](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677409562775.png)



- tensorboard可视化

代码演示：

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

zjc = zjc()
print(zjc)

input = torch.ones((64, 3, 32, 32))
output = zjc(input)
print(output.shape)

writer = SummaryWriter("seq_logs")
# 可视化计算图
writer.add_graph(model=zjc, input_to_model=input) # 这里要传入一个模型，一个传入模型的数据
writer.close()
```

运行结果：

![1677410088783](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677410088783.png)

![1677410165095](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677410165095.png)

![1677410263745](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677410263745.png)

#### 8.损失函数和反向传播

- 计算实际输出和目标之间的差距
- 为我们更新输出提供一定的依据(反向传播)



代码演示：

```python
import torch
import torchvision
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader

input = torch.tensor([1, 2, 3], dtype = torch.float32)
target = torch.tensor([1, 2, 5], dtype = torch.float32)

input = torch.reshape(input,(1,1,1,3))# 1batch_size,1channel,1行3列
target = torch.reshape(target,(1,1,1,3))

loss = L1Loss(reduction="sum") # 这里可以指定是求和还是取平均,默认是取平均

result = loss(input, target)
print(result)

# 平方差损失
mseloss = MSELoss()
result = mseloss(input, target)
print(result)

# 交叉熵损失

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1]) # 这里的y是指指向第二个类别
x = torch.reshape(x, (1, 3))
loss_cross = CrossEntropyLoss()
result_loss = loss_cross(x, y)
print(result_loss) # (-0.2)+ln(exp(0.1)+exp(0.2)+exp(0.3))
```

运行结果：

![1677470407468](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677470407468.png)



#### 9.优化器





代码演示：

```python
import torch.optim
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset,batch_size=1)
class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

zjc = zjc()
optim = torch.optim.SGD(zjc.parameters(), lr=0.01)
loss = CrossEntropyLoss()

for epoch in range(10):
    epoch_loss = 0
    for data in dataloader:
        img, target = data
        output = zjc(img)
        # print(output)
        # print(target)

        result_loss = loss(output, target)
        optim.zero_grad() # 先将参数置零
        result_loss.backward() # 将参数反向传播
        optim.step() # 梯度参数更新
        epoch_loss = epoch_loss + result_loss
    print(epoch_loss)

```

运行结果：

![1677470324955](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677470324955.png)



#### 10.现有模型的使用及修改

例如使用vgg16模型，对其进行修改



代码演示：

```python
import torchvision
from torch import nn
vgg16_false = torchvision.models.vgg16(pretrained = False)
# vgg16_true = torchvision.models.vgg16(pretrained = True)
print("ok")
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
# vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
# print(vgg16_true)
print(vgg16_false)
vgg16_false.classifier.add_module("add_linear",nn.Linear(1000,10))	#对模型增加一个线性层
# vgg16_false.classifier[7] = nn.Linear(4096, 10)
vgg16_false.classifier[6] = nn.Linear(4096, 10)	#对模型的某一层进行修改
print(vgg16_false)
```

运行结果：

![1677490923699](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677490923699.png)

![1677490993722](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677490993722.png)



#### 11.模型的保存和加载

- 模型的保存

演示代码：

```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式 1,模型结构 + 模型参数
torch.save(vgg16, "vgg16_method1.pth") # 填写保存模型的名字，和路径

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")# 保存模型的状态(参数)，用字典的形式，和路径
```

运行结果：会生成两个文件

![1677497410207](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677497410207.png)



- 模型加载

演示代码：

```python
import torch
import torchvision

# 方式1 -》保存方式1，加载模型
model = torch.load("vgg16_method1.pth") # 输入加载的模型存放的位置

print(model)

# 方式2 -》保存方式2，加载模型
# model = torch.load("vgg16_method2.pth")
# print(model)

# 如何还原成原来的模型呢
# 新建一个网络模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))

print(vgg16)
```





- 自定义模型保存（陷阱）

演示代码：

```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式 1,模型结构 + 模型参数
torch.save(vgg16, "vgg16_method1.pth") # 填写保存模型的名字，和路径

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")# 保存模型的状态(参数)，用字典的形式，和路径

# 陷阱
class Zjc(nn.Module):
    def __init__(self):
        super(Zjc, self).__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=3)

    def forward(self,x):
        x = self.conv1(x)
        return x

zjc = Zjc()

torch.save(zjc, "zjc_mothod1.pth") # 保存自定义模型
```

在另一个文件里加载时,会报错，这里是要让编译器能访问到你定义模型的方式

```python
model = torch.load("zjc_mothod1.pth")
print(model)
```

![1677497691952](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677497691952.png)



这个时候需要将模型的定义复制或者导入进来

```python
class Zjc(nn.Module):
    def __init__(self):
        super(Zjc, self).__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=3)

    def forward(self,x):
        x = self.conv1(x)
        return x

model = torch.load("zjc_mothod1.pth")
print(model)
```

或者：导入创建模型的那个文件

```python
from model_save import *
model = torch.load("zjc_mothod1.pth")
print(model)
```

#### 12.完整的模型训练套路

代码演示：

```python
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from zjc_model import *
# 读取数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度：{}".format(train_data_size))
print("测试数据集的长度：{}".format(test_data_size))

# 加载数据
trainloader = DataLoader(train_data, batch_size=64)
testloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络，一般将如下代码封装在一个单独的python文件中
# class zjc(nn.Module):
#     def __init__(self):
#         super(zjc, self).__init__()
#         self.model=nn.Sequential(
#             nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64*4*4, 64),
#             nn.Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x

# 创建网络模型
zjc = zjc()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# 还有一种写法是1e-2 = 1 x (10)^(-2) = 1/100 = 0.01
learning_rate = 0.01
optimizer = torch.optim.SGD(zjc.parameters(),lr=learning_rate)

total_train_step = 0
total_test_step = 0


# 训练次数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./modeltest_logs")

for i in range(epoch):
    print("第{}轮训练开始".format(i+1))

    # 训练步骤开始
    zjc.train() # 这句代码可加可不加，用于一些网络层有drop层的时候
    for data in trainloader:
        img, targets = data

        # 预测的输出
        output = zjc(img)
        loss = loss_fn(output, targets)

        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    zjc.eval() # 这句代码也是用于网络层中有drop层的，这里可以不加
    total_test_loss = 0
    total_accuracy = 0
    # 看模型在测试集上的效果
    with torch.no_grad(): # 只需要测试，不需要对梯度进行一个调整
        for data in testloader:
            imgs, targets=data
            output = zjc(imgs)
            accuracy_num = ((output.argmax(1) == targets).sum())
            total_accuracy = total_accuracy + accuracy_num
            loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + loss.item() # loss和loss.item()数值上没啥区别，只不过loss是tensor数据类型
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集的正确率：{}".format(total_accuracy/test_data_size)) # 正确的个数/测试集的数量
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存每一轮的模型
    torch.save(zjc, "zjc_{}.pth".format(i))
    # torch.save(zjc.state_dict(),"zjc_{}.pth".format(i))
    print("模型已保存")
writer.close()
```



一般模型会放在一个单独的文件夹里

```python
import torch
from torch import nn

# 搭建神经网络
class zjc(nn.Module):
    def __init__(self):
        super(zjc, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 测试模型是否正确
if __name__ == '__main__':
    zjc = zjc()
    input = torch.ones((64, 3, 32, 32))
    output = zjc(input)
    print(output.shape)
```





运行结果：

![1677552382518](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677552382518.png)

![1677552425807](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677552425807.png)

#### 13.利用GPU训练

- 方式一
  - 网络模型
  - 数据（输入，标注）
  - 损失函数
  - 以上可以使用.cuda()



演示代码：

```python
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 读取数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度：{}".format(train_data_size))
print("测试数据集的长度：{}".format(test_data_size))

# 加载数据
trainloader = DataLoader(train_data, batch_size=64)
testloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络，一般将如下代码封装在一个单独的python文件中
class Zjc(nn.Module):
    def __init__(self):
        super(Zjc, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
zjc = Zjc() # 变量名和类名尽量不要一样
# 网络模型使用gpu
if torch.cuda.is_available(): # 先判断cuda可不可用
    zjc = zjc.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 损失函数使用gpu
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
# 还有一种写法是1e-2 = 1 x (10)^(-2) = 1/100 = 0.01
learning_rate = 0.01
optimizer = torch.optim.SGD(zjc.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0


# 训练次数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./modeltest_logs")

for i in range(epoch):
    print("第{}轮训练开始".format(i+1))

    # 训练步骤开始
    zjc.train() # 这句代码可加可不加，用于一些网络层有drop层的时候
    for data in trainloader:
        img, targets = data
        # 数据使用gpu
        if torch.cuda.is_available():
            img = img.cuda()
            targets = targets.cuda()
        # 预测的输出
        output = zjc(img)
        loss = loss_fn(output, targets)

        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    zjc.eval() # 这句代码也是用于网络层中有drop层的，这里可以不加
    total_test_loss = 0
    total_accuracy = 0
    # 看模型在测试集上的效果
    with torch.no_grad(): # 只需要测试，不需要对梯度进行一个调整
        for data in testloader:
            imgs, targets=data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = zjc(imgs)
            accuracy_num = ((output.argmax(1) == targets).sum())
            total_accuracy = total_accuracy + accuracy_num
            loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + loss.item() # loss和loss.item()数值上没啥区别，只不过loss是tensor数据类型
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集的正确率：{}".format(total_accuracy/test_data_size)) # 正确的个数/测试集的数量
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存每一轮的模型
    torch.save(zjc, "zjc_{}.pth".format(i))
    # torch.save(zjc.state_dict(),"zjc_{}.pth".format(i))
    print("模型已保存")
writer.close()
```



运行结果：

![1677566571961](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677566571961.png)

![1677566616663](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677566616663.png)

![1677566699327](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677566699327.png)

![1677566726180](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677566726180.png)

- 使用google colab
  - 可以使用google免费提供的一个gpu
  - 前提是你要能访问谷歌，有谷歌账号
  - 使用起来类似于jupyter



- 方式2，更常用

![1677567312077](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677567312077.png)

代码演示：

```python
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 读取数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# device = torch.device("cpu")
# device = torch.device("cuda:0")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度：{}".format(train_data_size))
print("测试数据集的长度：{}".format(test_data_size))

# 加载数据
trainloader = DataLoader(train_data, batch_size=64)
testloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络，一般将如下代码封装在一个单独的python文件中
class Zjc(nn.Module):
    def __init__(self):
        super(Zjc, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
zjc = Zjc() # 变量名和类名尽量不要一样
# 网络模型使用gpu
# zjc.to(device)
zjc = zjc.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 损失函数使用gpu
# loss_fn.to(device)
loss_fn = loss_fn.to(device)
# 优化器
# 还有一种写法是1e-2 = 1 x (10)^(-2) = 1/100 = 0.01
learning_rate = 0.01
optimizer = torch.optim.SGD(zjc.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0


# 训练次数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./modeltest_logs")

for i in range(epoch):
    print("第{}轮训练开始".format(i+1))

    # 训练步骤开始
    zjc.train() # 这句代码可加可不加，用于一些网络层有drop层的时候
    for data in trainloader:
        img, targets = data
        # 数据使用gpu
        img = img.to(device)
        targets = targets.to(device)
        # 预测的输出
        output = zjc(img)
        loss = loss_fn(output, targets)

        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    zjc.eval() # 这句代码也是用于网络层中有drop层的，这里可以不加
    total_test_loss = 0
    total_accuracy = 0
    # 看模型在测试集上的效果
    with torch.no_grad(): # 只需要测试，不需要对梯度进行一个调整
        for data in testloader:
            imgs, targets=data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = zjc(imgs)
            accuracy_num = ((output.argmax(1) == targets).sum())
            total_accuracy = total_accuracy + accuracy_num
            loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + loss.item() # loss和loss.item()数值上没啥区别，只不过loss是tensor数据类型
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集的正确率：{}".format(total_accuracy/test_data_size)) # 正确的个数/测试集的数量
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存每一轮的模型
    torch.save(zjc, "zjc_{}.pth".format(i))
    # torch.save(zjc.state_dict(),"zjc_{}.pth".format(i))
    print("模型已保存")
writer.close()
```



#### 14.完整的模型验证

测试，demo的套路，利用已经训练好的数模型，然后，给它提供输入



```python
import torch
import torchvision
from PIL import Image
from torch import nn

# 导入图片
image_path = "./dataset/imgs/dog.png"
# image_path = "./dataset/imgs/airplane.png"
image = Image.open(image_path)

# 因为png格式是四个通道，除了RGB三通道外，还有一个透明度通道，保留其颜色通道
# 如果图片本来就是三个颜色通道，经过此操作，不变
# 加上这一步后，可以适应png，jpg各种格式的图片。因为不同截图软件截图保留的通道数是不一样的
image = image.convert("RGB")
# print(image)

# 调整图片的输入大小

# resize可以接收一个PIL类型返回PIL，或者接收tensor类型返回tensor类型
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

# 搭建神经网络
class Zjc(nn.Module):
    def __init__(self):
        super(Zjc, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
# 导入网络模型
model = torch.load("zjc_9.pth", map_location=torch.device("cpu")) # 因为加载的模型是在gpu上训练生成的，而此时运行代码用的是cpu，这里要将GPU上的东西映射到CPU上

# 神奇，这里为什么不用创建网络模型
# print(model)

image = torch.reshape(image, (1, 3, 32, 32))
# 这里养成良好习惯，习惯写下面格式的代码
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
```

各类别对应：

![1677573593430](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677573593430.png)

此时第10次模型的预测准确率在0.54左右

1. 这里我们给定的是一张小狗的图片

![1677573650434](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677573650434.png)

预测结果为：

![1677573819666](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677573819666.png)



误将小狗分类为鹿

2. 输入一张飞机的图片：

![1677573749022](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677573749022.png)

预测结果为

![1677573775601](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677573775601.png)

分类正确





#### 15.利用vgg16训练cifar10

vgg网络输入大小要求224x224

<font color='blue'>如果网络要求的输入的大小与实际的图像输入的大小不符，则改变等一个线性层的输入大小即可</font>



![1679975075597](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679975075597.png)



![1679975118223](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679975118223.png)

![1679919712007](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679919712007.png)





初试版本：

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

traindata = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
testdata = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_size = len(testdata)

traindataloader = DataLoader(dataset=traindata, batch_size=64)
testdataloader = DataLoader(dataset=testdata, batch_size=64)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)

        )
    def forward(self, x):
        x=self.model(x)
        return x
model = Model().to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

optim = torch.optim.SGD(model.parameters(), lr=0.01)

epoch = 30

train_step = 0
for i in range(epoch):
    print("第{}轮训练开始了".format(i+1))
    model.train()
    for data in traindataloader:
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)

        output = model(images)
        loss = loss_fn(output, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_step=train_step+1
        if train_step%500 == 0:
            print("第{}步训练时，训练集上的损失是{}".format(train_step, loss))

    model.eval()
    with torch.no_grad():
        test_loss = 0
        accuracy = 0
        for data in testdataloader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)
            loss = loss_fn(output, targets)
            test_loss += loss
            accuracy_num = (output.argmax(1) == targets).sum()
            accuracy += accuracy_num
    print("第{}轮上测试集的总损失是：{}".format(i+1, test_loss))
    print("第{}轮上的测试集的准确率为{}".format(i+1, accuracy/test_size))

                
```

![1679906132290](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679906132290.png)



batch_size=256时

![1680774052597](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1680774052597.png)

数据处理加了归一化操作后，batch_size同样是256

> torchvision.transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])

![1680774146090](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1680774146090.png)

改进之后：

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
transform_train = transforms.Compose(
    [transforms.Pad(4),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.RandomCrop(32, padding=4),
     ])

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)



traindata = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform = transform_train, download=True)
testdata = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform = transform_test, download=True)

test_size = len(testdata)

traindataloader = DataLoader(dataset=traindata, batch_size=64)
testdataloader = DataLoader(dataset=testdata, batch_size=64)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.ReLU(inplace=True), # inplace=True 表示对于输入的张量进行原地操作，即直接对原始的输入张量进行修改，而不是创建一个新的张量。这样做可以节省内存，但会覆盖原始的输入张量，可能会对后续的计算产生影响
            nn.Dropout(0.4),
            nn.Linear(64, 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)

        )
    def forward(self, x):
        x=self.model(x)
        return x
model = Model().to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

optim = torch.optim.SGD(model.parameters(), lr=0.01)

epoch = 20

train_step = 0
for i in range(epoch):
    print("第{}轮训练开始了".format(i+1))
    model.train()
    for data in traindataloader:
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)

        output = model(images)
        loss = loss_fn(images, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_step=train_step+1
        if train_step%500 == 0:
            print("第{}步训练时，训练集上的损失是{}".format(train_step, loss))

    model.eval()
    with torch.no_grad():
        test_loss = 0
        accuracy = 0
        for data in testdataloader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)
            loss = loss_fn(images, targets)
            test_loss += loss
            accuracy_num = (output.argmax(1) == targets).sum()
            accuracy += accuracy_num
    print("第{}轮上测试集的总损失是：{}".format(i+1, test_loss))
    print("第{}轮上的测试集的准确率为{}".format(i+1, accuracy/test_size))


```



完整版代码：

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
 
transform_train = transforms.Compose(
    [transforms.Pad(4),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.RandomCrop(32, padding=4),
])
 
transform_test = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
trainset = torchvision.datasets.CIFAR10(root='dataset_method_1', train=True, download=True, transform=transform_train)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True)
 
testset = torchvision.datasets.CIFAR10(root='dataset_method_1', train=False, download=True, transform=transform_test)
testLoader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=False)
 
vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
 
 
class VGG(nn.Module):
    def __init__(self, vgg):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg)
        self.dense = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(4096, 10)
 
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out
 
    def _make_layers(self, vgg):
        layers = []
        in_channels = 3
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
 
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
 
 
model = VGG(vgg)
# model.load_state_dict(torch.load('CIFAR-model/VGG16.pth'))
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-3)
loss_func = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4, last_epoch=-1)
 
 
total_times = 40
total = 0
accuracy_rate = []
 
 
def test():
    model.eval()
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images = images.to(device)
            outputs = model(images).to(device)
            outputs = outputs.cpu()
            outputarr = outputs.numpy()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    accuracy = 100 * correct / total
    accuracy_rate.append(accuracy)
    print(f'准确率为:{accuracy}%'.format(accuracy))
 
 
for epoch in range(total_times):
    model.train()
    model.to(device)
    running_loss = 0.0
    total_correct = 0
    total_trainset = 0
 
    for i, (data, labels) in enumerate(trainLoader, 0):
        data = data.to(device)
        outputs = model(data).to(device)
        labels = labels.to(device)
        loss = loss_func(outputs, labels).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        correct = (pred == labels).sum().item()
        total_correct += correct
        total_trainset += data.shape[0]
        running_loss += loss.item()
        if i % 1000 == 0 and i > 0:
            print(f"正在进行第{i}次训练, running_loss={running_loss}".format(i, running_loss))
            running_loss = 0.0
    test()
    scheduler.step()
 
 
# torch.save(model.state_dict(), 'CIFAR-model/VGG16.pth')
accuracy_rate = np.array(accuracy_rate)
times = np.linspace(1, total_times, total_times)
plt.xlabel('times')
plt.ylabel('accuracy rate')
plt.plot(times, accuracy_rate)
plt.show()
 

```



训练结果：准确率大大提高

![1679910463326](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679910463326.png)

#### 16.卷积

![1680490999159](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1680490999159.png) 

#### 17.逆置卷积

![1680497493730](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1680497493730.png)









#### 18 cifar10逐步提升准确率

初试版本：直接原始数据上来就做训练，训练epoch在20轮

使用的模型：

![1677407757242](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1677407757242.png)

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

traindata=torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
testdata=torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_size=len(testdata)

trainloader = DataLoader(traindata, batch_size=64)
testloader = DataLoader(testdata, batch_size=64)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
            
        )
        
    def forward(self, x):
        x=self.model(x)
        return x
    
net=Net().to(device)

loss_fn=nn.CrossEntropyLoss().to(device)

lr=0.01
optim=torch.optim.SGD(net.parameters(),lr=lr)

epoch=20

for epoch in range(epoch):
    print("训练开始")
    net.train()
    train_step=0
    for data in trainloader:
        images,targets=data
        images,targets=images.to(device),targets.to(device)
        
        output=net(images)
        loss=loss_fn(output,targets)
        train_step+=1
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if train_step%100==0:
            print("[epoch：{}，train_step:{},loss:{}]".format(epoch+1,train_step,loss))
            
        
    net.eval()
    with torch.no_grad():
        accuracy=0
        for data in testloader:
            images,targets=data
            images,targets=images.to(device),targets.to(device)
            
            output=net(images)
            accuracy+=(output.argmax(1)==targets).sum()
        print("本轮训练时，测试集的准确率是：{}".format(accuracy/test_size))
        
```

运行结果：第一轮和最后一轮测试集的准确率

![1681374661103](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681374661103.png)

![1681374688255](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681374688255.png)





改进1：全连接层由2层变为3层

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

traindata=torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
testdata=torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_size=len(testdata)

trainloader = DataLoader(traindata, batch_size=64)
testloader = DataLoader(testdata, batch_size=64)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,256),
            nn.Linear(256,64),
            nn.Linear(64,10)

        )

    def forward(self, x):
        x=self.model(x)
        return x

net=Net().to(device)

loss_fn=nn.CrossEntropyLoss().to(device)

lr=0.01
optim=torch.optim.SGD(net.parameters(),lr=lr)

epoch=20

for epoch in range(epoch):
    print("训练开始")
    net.train()
    train_step=0
    for data in trainloader:
        images,targets=data
        images,targets=images.to(device),targets.to(device)

        output=net(images)
        loss=loss_fn(output,targets)
        train_step+=1

        optim.zero_grad()
        loss.backward()
        optim.step()

        if train_step%100==0:
            print("[epoch：{}，train_step:{},loss:{}]".format(epoch+1,train_step,loss))


    net.eval()
    with torch.no_grad():
        accuracy=0
        for data in testloader:
            images,targets=data
            images,targets=images.to(device),targets.to(device)

            output=net(images)
            accuracy+=(output.argmax(1)==targets).sum()
        print("本轮训练时，测试集的准确率是：{}".format(accuracy/test_size))



```

运行结果：没多大提升

![1681374866431](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681374866431.png)

![1681374886230](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681374886230.png)

改进2：做了数据增强

```python
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 转化为tensor类型
    # 从[0,1]归一化到[-1,1]
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平镜像
    torchvision.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
    torchvision.transforms.RandomCrop(32, padding=4),  # 随机裁剪
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

```



完整代码：

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 转化为tensor类型
    # 从[0,1]归一化到[-1,1]
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平镜像
    torchvision.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
    torchvision.transforms.RandomCrop(32, padding=4),  # 随机裁剪
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

traindata=torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=train_transforms, download=True)
testdata=torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=test_transforms, download=True)

test_size=len(testdata)

trainloader = DataLoader(traindata, batch_size=64)
testloader = DataLoader(testdata, batch_size=64)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,256),
            nn.Linear(256,64),
            nn.Linear(64,10)

        )

    def forward(self, x):
        x=self.model(x)
        return x

net=Net().to(device)

loss_fn=nn.CrossEntropyLoss().to(device)

lr=0.005
optim=torch.optim.SGD(net.parameters(),lr=lr)

epoch=20

for epoch in range(epoch):
    print("训练开始")
    net.train()
    train_step=0
    for data in trainloader:
        images,targets=data
        images,targets=images.to(device),targets.to(device)

        output=net(images)
        loss=loss_fn(output,targets)
        train_step+=1

        optim.zero_grad()
        loss.backward()
        optim.step()

        if train_step%100==0:
            print("[epoch：{}，train_step:{},loss:{}]".format(epoch+1,train_step,loss))


    net.eval()
    with torch.no_grad():
        accuracy=0
        for data in testloader:
            images,targets=data
            images,targets=images.to(device),targets.to(device)

            output=net(images)
            accuracy+=(output.argmax(1)==targets).sum()
        print("本轮训练时，测试集的准确率是：{}".format(accuracy/test_size))

```

运行结果：提升也不是很明显：

![1681375115949](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681375115949.png)

![1681375141325](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681375141325.png)







增大epoch到40：运行结果

![1681472988175](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681472988175.png)

![1681473619126](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681473619126.png)













改进4：加深网络：

![1681375776063](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681375776063.png)

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 转化为tensor类型
    # 从[0,1]归一化到[-1,1]
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平镜像
    torchvision.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
    torchvision.transforms.RandomCrop(32, padding=4),  # 随机裁剪
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

traindata=torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=train_transforms, download=True)
testdata=torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=test_transforms, download=True)

test_size=len(testdata)

trainloader = DataLoader(traindata, batch_size=64)
testloader = DataLoader(testdata, batch_size=64)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model=nn.Sequential(
            # nn.Conv2d(3,32,5,1,2),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32,32,5,1,2),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32,64,5,1,2),
            # nn.MaxPool2d(2),
            # nn.Flatten(),
            # nn.Linear(64*4*4,256),
            # nn.Linear(256,64),
            # nn.Linear(64,10)

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ceil_mode=False),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ceil_mode=False),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ceil_mode=False),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ceil_mode=False),

            nn.Flatten(),  # 7 Flatten层
            nn.Linear(2048, 256),  # 8 全连接层
            nn.Linear(256, 64),  # 8 全连接层
            nn.Linear(64, 10)  # 9 全连接层



        )

    def forward(self, x):
        x=self.model(x)
        return x

net=Net().to(device)

loss_fn=nn.CrossEntropyLoss().to(device)

lr=0.005
optim=torch.optim.SGD(net.parameters(),lr=lr)

epoch=20

for epoch in range(epoch):
    print("训练开始")
    net.train()
    train_step=0
    for data in trainloader:
        images,targets=data
        images,targets=images.to(device),targets.to(device)

        output=net(images)
        loss=loss_fn(output,targets)
        train_step+=1

        optim.zero_grad()
        loss.backward()
        optim.step()

        if train_step%100==0:
            print("[epoch：{}，train_step:{},loss:{}]".format(epoch+1,train_step,loss))


    net.eval()
    with torch.no_grad():
        accuracy=0
        for data in testloader:
            images,targets=data
            images,targets=images.to(device),targets.to(device)

            output=net(images)
            accuracy+=(output.argmax(1)==targets).sum()
        print("本轮训练时，测试集的准确率是：{}".format(accuracy/test_size))

```



运行结果：效果提升比较明显，但epoch太小了，还未收敛

![1681375252991](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681375252991.png)

![1681375291139](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681375291139.png)



epoch增加到40后

![1681475554845](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681475554845.png)



![1681475574904](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681475574904.png)





改进5：使用残差网络，效果也还可以，可能epoch设置太小不太明显

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 转化为tensor类型
    # 从[0,1]归一化到[-1,1]
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平镜像
    torchvision.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
    torchvision.transforms.RandomCrop(32, padding=4),  # 随机裁剪
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

traindata=torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=train_transforms, download=True)
testdata=torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=test_transforms, download=True)

test_size=len(testdata)

trainloader = DataLoader(traindata, batch_size=64)
testloader = DataLoader(testdata, batch_size=64)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Conv_Block(nn.Module):

    def __init__(self, inchannel, outchannel, res=True):
        super(Conv_Block, self).__init__()
        self.res = res  # 是否带残差连接
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(outchannel))
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, res=True):
        super(Net, self).__init__()

        self.block1 = Conv_Block(inchannel=3, outchannel=64)
        self.block2 = Conv_Block(inchannel=64, outchannel=128)
        self.block3 = Conv_Block(inchannel=128, outchannel=256)
        self.block4 = Conv_Block(inchannel=256, outchannel=512)
        # 构建卷积层之后的全连接层以及分类器：

        self.classifier = nn.Sequential(nn.Flatten(),  # 7 Flatten层
                                     nn.Dropout(0.4),
                                     nn.Linear(2048, 256),  # 8 全连接层
                                     nn.Linear(256, 64),  # 8 全连接层
                                     nn.Linear(64, 10))  # 9 全连接层 )   # fc，最终Cifar10输出是10类

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=2))  # 1最大池化层

    def forward(self, x):
        out = self.block1(x)
        out = self.maxpool(out)
        out = self.block2(out)
        out = self.maxpool(out)
        out = self.block3(out)
        out = self.maxpool(out)
        out = self.block4(out)
        out = self.maxpool(out)
        out = self.classifier(out)
        return out



net=Net().to(device)

loss_fn=nn.CrossEntropyLoss().to(device)

lr=0.005
optim=torch.optim.SGD(net.parameters(),lr=lr)

epoch=20

for epoch in range(epoch):
    print("训练开始")
    net.train()
    train_step=0
    for data in trainloader:
        images,targets=data
        images,targets=images.to(device),targets.to(device)

        output=net(images)
        loss=loss_fn(output,targets)
        train_step+=1

        optim.zero_grad()
        loss.backward()
        optim.step()

        if train_step%100==0:
            print("[epoch：{}，train_step:{},loss:{}]".format(epoch+1,train_step,loss))


    net.eval()
    with torch.no_grad():
        accuracy=0
        for data in testloader:
            images,targets=data
            images,targets=images.to(device),targets.to(device)

            output=net(images)
            accuracy+=(output.argmax(1)==targets).sum()
        print("本轮训练时，测试集的准确率是：{}".format(accuracy/test_size))
```

![1681375406054](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681375406054.png)

![1681375464246](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1681375464246.png)

