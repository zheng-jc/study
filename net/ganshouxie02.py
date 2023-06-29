import torch.utils.data
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

tf = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
])
traindata = torchvision.datasets.MNIST(root='./dataset', train=True, transform=tf, download=True)
testdata = torchvision.datasets.MNIST(root='./dataset', train=False, transform=tf, download=True)

dataset = torch.utils.data.ConcatDataset([traindata, testdata])
print(len(dataset))

batch_size = 128
latent_dim = 100


dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.gen(x)
        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.dis(x)
        return x

gen = generator().to(device)
dis = discriminator().to(device)

loss_fn = nn.BCELoss().to(device)

g_optim = torch.optim.Adam(gen.parameters(), lr=0.0003)
d_optim = torch.optim.Adam(dis.parameters(), lr=0.0003)

epochs = 100

# 第二个参数是维度大小
label_ones = torch.ones(batch_size, 1).to(device)
label_zeros = torch.zeros(batch_size, 1).to(device)

for epoch in range(epochs):
    train_step = 0
    print("运行开始")
    for data in dataloader:
        train_step += 1
        gt_image, _ = data
        gt_image = gt_image.to(device)
        z = torch.randn(batch_size, latent_dim).to(device)
        pred_image = gen(z)
        # real_image = image.view(784, -1)

        real_out = dis(gt_image)
        fake_out = dis(pred_image)

        real_scores = real_out
        fake_scores = fake_out

        real_loss = loss_fn(real_out, label_ones)
        fake_loss = loss_fn(fake_out, label_zeros)
        d_loss = (real_loss+fake_loss)

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        z = torch.randn(batch_size, latent_dim).to(device)
        pred_images = gen(z)
        fake_out = dis(pred_images)
        g_loss = loss_fn(fake_out, label_ones)

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        if train_step % 100 == 0:
            print("[{}/{}]:D_loss:{},G_loss:{},D_real:{},D_fake:{}".format(epoch, epochs,
                                                                           d_loss.item(), g_loss.item(),
                                                                           real_scores.data.mean(), fake_scores.data.mean()))

        if epoch == 0:
            real_images = make_grid(gt_image, nrow=8, padding=2)
            torchvision.utils.save_image(real_images, './img/real_images.png')

        # print(pred_image.shape) # torch.Size([128, 784])
        pred_image = pred_image.view(128,1,28,28) #(N,C,H,W)
        # print(pred_image.shape) # torch.Size([128, 1, 28, 28])
        fake_images = make_grid(pred_image, nrow=8, padding=2)
        torchvision.utils.save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))
    print("运行结束")

torch.save(gen.state_dict(), './gen_shouxie.pth')
torch.save(dis.state_dict(), './dis_shouxie.pth')