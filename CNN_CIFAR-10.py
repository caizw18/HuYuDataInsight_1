#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


import torch.nn as nn
import torch.nn.functional as F


# In[20]:


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    
    def forward(self,x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))
        
        x = x.view(-1,x.size()[1:].numel())
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# In[21]:


net = Net()
print(net)


# In[5]:


input = torch.randn(1,1,32,32)
print(input)


# In[6]:


out = net(input)
print(out)


# # CIFAR-10
# - 使用torch的torchvision加载初始化数据集
# - 定义卷积神经网络
# - 定义损失函数
# - 根据训练数据训练网络
# - 测试数据上测试网络

# In[7]:


import torch
import torchvision
import torchvision.transforms as transforms


# ## Dataset
# 定义好数据的格式和数据变换形式
# ## Dataloader
# 用iterative的方式不断读入批次数据

# In[28]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    
])

trainset = torchvision.datasets.CIFAR10(root='./Data',train=True,
                                       download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./Data',train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,
                                         shuffle=True,num_workers=0)

testloader = torch.utils.data.DataLoader(testset,batch_size=4,
                                         shuffle=False,num_workers=0)


# In[10]:


classes = ("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","trunk")


# In[13]:


import matplotlib.pyplot as plt
import numpy as np
#tensor[batch,channel,H,W]
#H,W,CHANNEL
def imshow(img):  #恢复为正常图片然后通道转换再显示
    img = img/2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    
dataiter = iter(trainloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print(labels)
print(classes[labels[0]])


# In[24]:


criterion = nn.CrossEntropyLoss()


# In[25]:


import torch.optim as optim
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)


# In[26]:


#epoch学习次数
for epoch in range(2):
    
    running_loss = 0.0
    #4批信息，对每一批
    for i, data in enumerate(trainloader,0):
        
        inputs, labels = data
        
        #梯度清零
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 2000 == 1999:
            print('[%d,%5d] loss:%.3f' % (epoch+1,i+1,running_loss/2000))
            running_loss = 0.0

print("Finish")


# In[22]:


print(net)


# In[27]:


PATH = './cifar_net.pth'
torch.save(net.state_dict(),PATH)


# In[29]:


dataiter = iter(testloader)
#迭代一次
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:',' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[30]:


net = Net()
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))


# In[31]:


outputs = net(images)


# In[32]:


print(outputs)


# In[33]:


#找行最大值的最大位置
_, predicted = torch.max(outputs,1)
print('Predicted',' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# In[35]:


#总的预测效果
correct = 0
total = 0
#预测时关闭梯度
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


# In[37]:


print(correct/total)


# In[ ]:




