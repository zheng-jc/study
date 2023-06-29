# import torch
# import torchvision
# from matplotlib import pyplot as plt
#
# import torch.autograd
# from torch import nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision.utils import  save_image
#
#
# def to_img(x):
#     out=0.5*(x+1)
#     out=out.clamp(0,1)#Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
#     out=out.view(-1,1,28,28)#view()函数作用是将一个多行的Tensor,拼接成一行
#     return out
# batch_size=128
# num_epoch=100
# z_dimension=100
#
# #图形啊处理过程
# img_transform=torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=[0.5],std=[0.5])
# ])
# traindata=torchvision.datasets.MNIST(root='./dataset', download=True, train=True, transform=img_transform)
# # testdata=torchvision.datasets.MNIST(root='./dataset', train=False, download=True,transform=torchvision.transforms.ToTensor())
# # test_size=len(testdata)
#
# # image, label=testdata[0]
# # image.show()
#
# dataloader=DataLoader(traindata,batch_size=batch_size,shuffle=True)
# # testloader=DataLoader(testdata,batch_size=128)
#
# # 判别器
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.dis=nn.Sequential(
#             nn.Linear(784,256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256,256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256,1),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         x=self.dis(x)
#         return x.squeeze(-1)
#
# # 生成器
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.gen=nn.Sequential(
#             nn.Linear(100,256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256,256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256,784),
#             nn.Tanh()# tanh()使结果在[-1,1]之间
#         )
#     def forward(self,x):
#         x=self.gen(x)
#         return x
#
# D=Discriminator()
# G=Generator()
# if torch.cuda.is_available():
#     D=D.cuda()
#     G=G.cuda()
#
#
# criterion=nn.BCELoss()
# d_optimizer=torch.optim.Adam(D.parameters(),lr=0.0003)
# g_optimizer=torch.optim.Adam(G.parameters(),lr=0.0003)
#
#
#
# for epoch in range(num_epoch):
#     for i,(img,_) in enumerate(dataloader):
#         num_img=img.size(0)
#
#         img=img.view(num_img,-1)
#         real_img=Variable(img).cuda()
#         real_label=Variable(torch.ones(num_img)).cuda()
#         fake_label=Variable(torch.zeros(num_img)).cuda()
#
#         # 计算真实图片的损失
#         real_out=D(real_img)
#         d_loss_real=criterion(real_out,real_label)
#         real_scores=real_out
#
#         # 计算假图片的真实值
#         z=Variable(torch.randn(num_img,z_dimension)).cuda()
#         fake_img=G(z)
#         fake_out=D(fake_img)
#         d_loss_fake=criterion(fake_out,fake_label)
#         fake_scores=fake_out
#
#         # 损失函数和优化
#         d_loss=d_loss_fake+d_loss_real
#         d_optimizer.zero_grad()
#         d_loss.backward()
#         d_optimizer.step()
#
#         # 训练生成器
#
#         # 计算假图片的损失
#         z=Variable(torch.randn(num_img,z_dimension)).cuda()
#         fake_img=G(z)
#         output=D(fake_img)
#         g_loss=criterion(output,real_label)
#
#         # 反向传播和优化
#         g_optimizer.zero_grad()
#         g_loss.backward()
#         g_optimizer.step()
#
#         # 打印中间的损失
#         if (i + 1) % 100 == 0:
#             print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
#                   'D real: {:.6f},D fake: {:.6f}'.format(
#                 epoch, num_epoch, d_loss.data[0], g_loss.data[0],
#                 real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
#             ))
#
#         if epoch == 0:
#             real_images = to_img(real_img.cpu().data)
#             save_image(real_images, './img/real_images.png')
#
#         fake_images = to_img(fake_img.cpu().data)
#         save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))
#     print("训练结束")
#     # 保存模型
#     torch.save(G.state_dict(), './modelsave/generator.pth')
#     torch.save(D.state_dict(), './modelsave/discriminator.pth')
#
#


# coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 创建文件夹
if not os.path.exists('./img'):
    os.mkdir('./img')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out


batch_size = 128
num_epoch = 100
z_dimension = 100

# 图形啊处理过程
img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    transforms.Normalize((0.5,),(0.5,)),
])

# mnist dataset mnist数据集下载
mnist = datasets.MNIST(
    root='./dataset', train=True, transform=img_transform, download=True
)

# data loader 数据载入
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True
)


# 定义判别器  #####Discriminator######使用多层网络来作为判别器

# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )

    def forward(self, x):
        x = self.dis(x)
        x = x.squeeze(-1)
        return x


####### 定义生成器 Generator #####
# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 784),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间
        )

    def forward(self, x):
        x = self.gen(x)
        x = x.squeeze(-1)
        return x



# 创建对象
D = discriminator()
G = generator()
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

#########判别器训练train#####################
# 分为两部分：1、真的图像判别为真；2、假的图像判别为假
# 此过程中，生成器参数不断更新

# 首先需要定义loss的度量方式  （二分类的交叉熵）
# 其次定义 优化函数,优化函数的学习率为0.0003
criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

###########################进入训练##判别器的判断过程#####################

for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # view()函数作用是将一个多行的Tensor,拼接成一行
        # 第一个参数是要拼接的tensor,第二个参数是-1
        # =============================训练判别器==================
        img = img.view(num_img, -1)  # 将图片展开为28*28=784
        real_img = Variable(img).cuda()  # 将tensor变成Variable放入计算图中
        real_label = Variable(torch.ones(num_img)).cuda()  # 定义真实的图片label为1
        fake_label = Variable(torch.zeros(num_img)).cuda()  # 定义假的图片的label为0

        # 计算真实图片的损失
        real_out = D(real_img)  # 将真实图片放入判别器中
        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
        real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好

        # 计算假的图片的损失
        z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 随机生成一些噪声
        fake_img = G(z)  # 随机噪声放入生成网络中，生成一张假的图片
        fake_out = D(fake_img)  # 判别器判断假的图片
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好

        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数

        # ==================训练生成器============================
        ################################生成网络的训练###############################
        # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
        # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
        # 反向传播更新的参数是生成网络里面的参数，
        # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
        # 这样就达到了对抗的目的

        # 计算假的图片的损失

        z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 得到随机噪声
        fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
        output = D(fake_img)  # 经过判别器得到的结果
        g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss

        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

        # 打印中间的损失
        if (i + 1) % 100 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                # epoch, num_epoch, d_loss.data[0], g_loss.data[0],
                epoch, num_epoch, d_loss.item(), g_loss.item(),
                real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
            ))

        if epoch == 0:
            real_images = to_img(real_img.cpu().data)
            save_image(real_images, './img/real_images.png')

        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))
    print("运行结束")
# 保存模型
torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')
