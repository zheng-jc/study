import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


import argparse

parser=argparse.ArgumentParser(description='this is a demo')
parser.add_argument("--epochs",type=int,default=100,help="number of epoch of training")
parser.add_argument("--batch_size",type=int,default=128,help="size of the batch")
parser.add_argument("--lr",type=float,default=0.0003,help="CrossEntropyLoss:learning rate")
parser.add_argument("--latent_dim",type=int,default=100,help="Dimension of noise")

opt=parser.parse_args()
print(opt)


img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    torchvision.transforms.Normalize((0.5,),(0.5,))
    # torchvision.transforms.Normalize((0.1307,),(0.3081,))
])
traindata=torchvision.datasets.MNIST(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# testdata=torchvision.datasets.MNIST(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)
# print(len(traindata)) #60000条数据
# print(len(testdata))



# batch_size=128 # 因为60000条训练数据，除以128除不尽，最后一个批量为96，而此时BCELoss的输入是（128,1）的label_one和(96,1)的判别器输出，所以报错，但是将批量大小改为32时不会报错
# latent_dim=100
# dataloader=DataLoader(traindata,batch_size=batch_size,shuffle=True) # 这里batch_size为128时出现了错误，但是将最后一个不满足128批量的数据丢弃后，错误解决
dataloader=DataLoader(traindata,batch_size=opt.batch_size,shuffle=True,drop_last=True)


def to_img(x):
    out = 0.5 * (x + 1)
    # out = 0.3081 * x + 0.1307
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Geneator(nn.Module):
    def __init__(self):
        super(Geneator, self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,784),
            nn.Tanh()
        )
    def forward(self,x):
        x=self.gen(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(784,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=x.view(-1,784)
        x=self.dis(x)
        return x

gen=Geneator().to(device)
dis=Discriminator().to(device)

loss_fn=nn.BCELoss().to(device)


D_optim=torch.optim.Adam(dis.parameters(),lr=opt.lr)
G_optim=torch.optim.Adam(gen.parameters(),lr=opt.lr)

# epochs=100

label_ones=torch.ones(opt.batch_size,1).to(device)
label_zeros=torch.zeros(opt.batch_size,1).to(device)

for epoch in range(opt.epochs):
    train_step=-1
    print("运行开始")
    for data in dataloader:

        # 这样就可以从0开始了
        train_step+=1
        gt_image,_=data
        gt_image=gt_image.to(device)
        z = torch.randn(opt.batch_size,opt.latent_dim).to(device)
        pred_images=gen(z)

        real_image=gt_image.view(784,-1)


        real_out=dis(gt_image)
        # print(real_out.size())
        fake_out=dis(pred_images)

        real_scores=real_out
        fake_scores=fake_out

        # 先训练判别器
        D_optim.zero_grad()
        # print(real_out.shape)
        # print(label_ones.shape)
        real_loss = loss_fn(real_out, label_ones)
        fake_loss = loss_fn(fake_out, label_zeros)
        D_loss = real_loss + fake_loss
        D_loss.backward()
        D_optim.step()

        # 再训练生成器
        z = torch.randn(opt.batch_size, opt.latent_dim).to(device)
        pred_images = gen(z)
        fake_out=dis(pred_images)
        G_optim.zero_grad()
        # G_loss=loss_fn(dis(pred_images.detach()),label_ones)

        # 这是新噪声，所以不需要对梯度进行处理
        G_loss=loss_fn(dis(pred_images),label_ones)
        G_loss.backward()
        G_optim.step()
        if (train_step + 1) % 100 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                # epoch, num_epoch, d_loss.data[0], g_loss.data[0],
                epoch, opt.epochs, D_loss.item(), G_loss.item(),
                real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
            ))
        if epoch == 0:
            real_images = to_img(real_image.cpu().data)
            torchvision.utils.save_image(real_images, './img/real_images.png')

        fake_images = to_img(pred_images.cpu().data)
        torchvision.utils.save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))
    print("运行结束")


torch.save(gen.state_dict(), './gen_shouxie.pth')
torch.save(dis.state_dict(), './dis_shouxie.pth')




