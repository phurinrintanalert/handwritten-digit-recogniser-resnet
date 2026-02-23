import torch
import torch.nn as nn
import \
    torch.nn.functional as F  # contains some useful functions like activation functions & convolution operations you can use

import torchvision
import numpy as np
from altair.vegalite.v6.schema import channels
from torchvision import datasets, models, transforms

class ResidualBlock(nn.Module):

    def __init__(self, channels1, channels2, res_stride=1):
        super(ResidualBlock, self).__init__()
        self.inplanes = channels1
        # Exercise 2.1.1 construct the block without shortcut
        self.conv1 = nn.Conv2d(channels1,channels2,kernel_size = 3, stride = res_stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = channels2)
        self.conv2 = nn.Conv2d(channels2,channels2,kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = channels2)

        if res_stride != 1 or channels2 != channels1:
            # Exercise 2.1.3 the shortcut; create option for resizing input
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels1,channels2,kernel_size = 1, stride = res_stride, bias = False),
                nn.BatchNorm2d(num_features = channels2)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):

        # forward pass: Conv2d > BatchNorm2d > ReLU > Conv2D >  BatchNorm2d > ADD > ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # THIS IS WHERE WE ADD THE INPUT
        #print('input shape',x.shape,self.inplanes)
        out += self.shortcut(x)
        #print('res block output shape',  out.shape)
        # final ReLu
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_strides, num_features, in_channels, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # step 1. Initialising the network with a 3 x3 conv and batch norm
        self.conv1 = nn.Conv2d(in_channels, num_features[0], kernel_size=3, stride=num_strides[0], padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Step 2: TO DO Using function make_layer() create 4 residual layers
        # num_blocks per layer is given by input argument num_blocks (which is an array)
        self.layer1 = self._make_layer(block, num_features[1], num_blocks, num_strides[1])
        self.layer2 = self._make_layer(block,num_features[2], num_blocks, num_strides[2])
        self.layer3 = self._make_layer(block,num_features[3], num_blocks, num_strides[3])
        self.layer4 = self._make_layer(block,num_features[4], num_blocks, num_strides[4])
        self.linear = nn.Linear(num_features[4], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []

        for i in np.arange(num_blocks - 1):
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes

        layers.append(block(planes, planes, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


import torch.optim as optim

def my_ResNet4(in_channels=1):
    return ResNet(ResidualBlock,2, [1,1,2,2,2], [64,64,128,256,512], in_channels=in_channels)

if __name__ == '__main__':

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # This is used to transform the images to Tensor and normalize it
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    training = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(training, batch_size=8,
                                               shuffle=True)

    testing = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testing, batch_size=8,
                                              shuffle=False)

    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')

    resnet = my_ResNet4(in_channels=1).to(device)

    loss_fun = nn.CrossEntropyLoss()

    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    epochs = 1
    for epoch in range(epochs):
        resnet.train()
        # enumerate can be used to output iteration index i, as well as the data
        for i, (data, labels) in enumerate(train_loader, 0):
            data,labels = data.to(device),labels.to(device)
            optimizer.zero_grad()
            out = resnet(data)
            loss = loss_fun(out,labels)
            loss.backward()
            optimizer.step()
            # print statistics
            ce_loss = loss.item()
            if i % 1000 == 0:
                print('[%d, %5d] loss: %.3f' %
                     (epoch + 1, i + 1, ce_loss))

    resnet.eval()
    correct_guess = 0
    total = 0

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader, 0):
            data,labels = data.to(device),labels.to(device)
            y_score = resnet(data)
            _,y_predict = torch.max(y_score, 1)
            total += labels.size(0)
            correct_guess += torch.eq(y_predict, labels).sum().item()

    print(f"accuracy on test set: {100 * correct_guess / total}")

    torch.save(resnet.state_dict(), "mnist_model.pth")
