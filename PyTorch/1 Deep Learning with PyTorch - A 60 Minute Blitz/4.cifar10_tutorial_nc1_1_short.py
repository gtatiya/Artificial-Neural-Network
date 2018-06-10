
# coding: utf-8

# In[1]:

import torch
import torchvision
import torchvision.transforms as transforms
import time


# In[2]:

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # GT: why Normalize??

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 2. Define a Convolution Neural Network
# --------------------------------------
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).
# 
# 

# In[3]:

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 instead of 1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# 3. Define a Loss function and optimizer
# ---------------------------------------
# 
# Let's use a Classification Cross-Entropy loss and SGD with momentum

# In[4]:

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 4. Train the network
# --------------------
# 
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize

# In[11]:

start_time = time.time()
file = open('NetworkConfig_1.txt', 'w')
file.close()

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            # writing to a text file
            file = open('NetworkConfig_1.txt', 'a') # append to the file created
            s = "["+str(epoch + 1)+", "+str(i + 1)+"] loss: "+str(running_loss / 2000)
            running_loss = 0.0
            
            print("Time: ", (time.time()-start_time)/(60*60), " hr")
            s += " --> "+"Time: "+str((time.time()-start_time)/(60*60))+" hr\n"
            
            file.write(s)
            file.close()

print('Finished Training')
end_time = time.time()
print("The network took ", (time.time()-start_time)/(60*60), " hr to train")


# 5. Test the network on the test data
# ------------------------------------
# 
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
# 
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
# 
# Okay, first step. Let us display an image from the test set to get familiar.

# In[12]:

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# writing to a text file
file = open('NetworkConfig_1.txt', 'a') # append to the file created
s = "\n\n\n Accuracy of the network on the 10000 test images: "+str(100 * correct / total)+"\n\n\n"
file.write(s)
file.close()


# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
# 
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:
# 
# 

# In[14]:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze() # c will have 4, (0 or 1) values
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

    # writing to a text file
    file = open('NetworkConfig_1.txt', 'a') # append to the file created
    s = "Accuracy of "+str(classes[i])+" "+str(100 * class_correct[i] / class_total[i])+"\n"
    file.write(s)
    file.close()


# In[ ]:



