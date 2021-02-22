import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import torch.utils.data as Data
#第一步，构建一个dataloader   
#read npy
x_train = np.load("path").astype("float32") #(n,img_w,img_h,rgb)
y_train = np.load("path")
 #(n,1)  ---> (n,1,10)

#x_test 
#y_test


# 创建一个转换器，将torchvision数据集的输出范围[0,1]转换为归一化范围的张量[-1,1]。
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   
x_train = torch.Tensor(np.array([transform(img).numpy() for img in x_train]))
num_classes = 10
y_train = y_train.type(torch.int64)
y_train = torch.zeros(y_train.shape[0],10).scatter_(1,y_train,1)

# 创建训练集
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
train_dataset = Data.TensorDataset(x_train,y_train)
#x_train: dataset   normal   shape (sample,img_w,img_h,rgb)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=5,shuffle=True, num_workers=2)

# 创建测试集
#testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(x_test, batch_size=5,shuffle=False, num_workers=2)

# 类别标签
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#第二步，导入模型，放入gpu 
from net.netcode import Net

net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net.to(device)

#第二步，声明一些函数，loss funcation optimizer 
import torch.optim as optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



#第三步，训练循环，用gpu 
for epoch in range(2):  # loop over the dataset multiple times              #整个数据训练两轮

    #所有数据开始训练  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)


        #经典五步
       
        optimizer.zero_grad()                    #将梯度初始化为零

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)        # 计算loss
        loss.backward()                          # loss 求导
        optimizer.step()                         # 更新参数


        #输出损失函数 两千轮显示一次
        # print statistics
        running_loss += loss.item()              #获取tensor的数值
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0                   # 每2000次迭代，输出loss的平均值

print('Finished Training')



#第四步，保存模型到./model
PATH = './model'
torch.save(net.state_dict(), PATH)