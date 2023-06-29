import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

traindata=torchvision.datasets.MNIST(root='./dataset', transform=torchvision.transforms.ToTensor(), train=True, download=True)
testdata=torchvision.datasets.MNIST(root='./dataset',transform=torchvision.transforms.ToTensor(),train=False,download=True)

testdata_size=len(testdata)
traindataloader=DataLoader(traindata,batch_size=128,shuffle=True,drop_last=True)
testdataloader=DataLoader(testdata,batch_size=128,shuffle=True,drop_last=True)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TeacherModel(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(784,1200)
        self.fc2=nn.Linear(1200,1200)
        self.fc3=nn.Linear(1200,num_classes)
        self.dropout=nn.Dropout(p=0.5)

    def forward(self, x):
        x=x.view(-1,784)
        x=self.fc1(x)
        x=self.dropout(x)
        x=self.relu(x)

        x=self.fc2(x)
        x=self.dropout(x)
        x=self.relu(x)

        x=self.fc3(x)

        return x

teacher=TeacherModel().to(device)

criterion=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(teacher.parameters(),lr=1e-4)

epochs=10

print("教师模型")
for epoch in range(epochs):
    teacher.train()
    # print("epoch{}训练开始".format(epoch+1))
    for data in traindataloader:
        images,targets=data
        images,targets=images.to(device),targets.to(device)

        pred=teacher(images)

        loss=criterion(pred,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    teacher.eval()
    num_accuracy=0
    with torch.no_grad():
        for data in testdataloader:
            images,targets=data
            images,targets=images.to(device),targets.to(device)
            pred=teacher(images)

            num_accuracy+=((pred.argmax(1)==targets).sum())
    print("[{}/{}]:teacher testdata_acccuracy:{}".format(epoch+1,epochs,num_accuracy/testdata_size))

torch.save(teacher,"teacher.pth")

class StudentModel(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(StudentModel, self).__init__()
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(784,20)
        self.fc2=nn.Linear(20,20)
        self.fc3=nn.Linear(20,num_classes)
        self.dropout=nn.Dropout(p=0.5)

    def forward(self, x):
        x=x.view(-1,784)
        x=self.fc1(x)
        # x=self.dropout(x)
        x=self.relu(x)

        x=self.fc2(x)
        # x=self.dropout(x)
        x=self.relu(x)

        x=self.fc3(x)

        return x

student1=StudentModel().to(device)

criterion=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(student1.parameters(),lr=1e-4)

epochs=10
print("学生模型1")
for epoch in range(epochs):
    student1.train()
    # print("epoch{}训练开始".format(epoch+1))
    for data in traindataloader:
        images,targets=data
        images,targets=images.to(device),targets.to(device)

        pred=student1(images)

        loss=criterion(pred,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    student1.eval()
    num_accuracy=0
    with torch.no_grad():
        for data in testdataloader:
            images,targets=data
            images,targets=images.to(device),targets.to(device)
            pred=student1(images)

            num_accuracy+=((pred.argmax(1)==targets).sum())
    print("[{}/{}]:student1 testdata_acccuracy:{}".format(epoch+1,epochs,num_accuracy/testdata_size))


print("知识蒸馏后的学生模型2")

student2=StudentModel().to(device)

hard_loss=nn.CrossEntropyLoss().to(device)
soft_loss=nn.KLDivLoss(reduction="batchmean").to(device)
optimizer=torch.optim.Adam(student2.parameters(),lr=1e-4)

temp=7
alpha=0.3

teacher1=torch.load("teacher.pth")
epochs=10
for epoch in range(epochs):
    student2.train()
    for data in traindataloader:
        images,targets=data
        images,targets=images.to(device),targets.to(device)

        teacher.eval()
        with torch.no_grad():
            teacher_pred=teacher1(images)

        student2_pred=student2(images)

        # 这个损失是学生预测和标准答案之间的损失
        student_loss=hard_loss(student2_pred,targets)
        # 这个损失是学生蒸馏和老师蒸馏后之间的损失
        ditillation_loss=soft_loss(F.softmax(student2_pred/temp,dim=1),
                                   F.softmax(teacher_pred/temp,dim=1))

        # 将hard_loss和soft_loss加权求和
        loss = alpha * student_loss + (1-alpha) * ditillation_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    student2.eval()
    num_accuracy=0
    with torch.no_grad():
        for data in testdataloader:
            images,targets=data
            images,targets=images.to(device),targets.to(device)
            pred=student2(images)

            num_accuracy+=((pred.argmax(1)==targets).sum())

    print("[{}/{}]:student2 testdata_acccuracy:{}".format(epoch+1,epochs,num_accuracy/testdata_size))
