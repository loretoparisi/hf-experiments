# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import matplotlib
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# MLP models
from res_mlp_pytorch.res_mlp_pytorch import ResMLP
from mlp_mixer.mlp_mixer import MLPMixer

# LP: to png
matplotlib.use('agg')

# Image normalize
transform = T.Compose(
    [T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Image resize 256
transform256 = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

cache_dir = os.getenv("cache_dir", "../../models")

# choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Res MLP
res_model = ResMLP(
    image_size = 224,
    patch_size = 16,
    dim = 512,
    depth = 12,
    num_classes = 1000
)
res_model.to(torch.device(device))

# MLP Mixer
mixer_model = MLPMixer(in_channels=3, 
                image_size=224, 
                patch_size=16, 
                num_classes=1000,
                dim=512, 
                depth=8, 
                token_dim=256, 
                channel_dim=2048)
mixer_model.to(torch.device(device))

class CNN(nn.Module):
    '''
        Naive CNN network
    '''
    def __init__(self):
        super(CNN, self).__init__()

        # FeedForward(dim, channel_dim, dropout)
        
        #image_size=224, 
        #patch_size=16, 
        #num_classes=1000,
        #dim=512, 
        #depth=8, 
        #token_dim=256, 
        #channel_dim=2048
        #dropout=0.
        
        self.in_channels=3
        self.dim = 120
        self.hidden_dim = 84
        self.patch_size = 16
        
        self.image_size = 224
        self.token_dim = 256
        self.num_patch = (self.image_size// self.patch_size) ** 2

        self.conv1 = nn.Conv2d(self.in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, self.patch_size, 5)
        self.fc1 = nn.Linear(self.patch_size * 5 * 5, self.dim)
        self.fc2 = nn.Linear(self.dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.patch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

cnn_model = CNN()
cnn_model.to(torch.device(device))

def train(model, device, dataloader, cache_dir=''):
    '''
        Naive image classification training code
    '''
    # loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(res_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):

            inputs, labels = data                         # this is what you had
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(model.state_dict(), os.path.join(cache_dir, 'cifar.pt'))
    print('model training ended')

def test(model, device, dataloader, cache_dir=''):
    '''
        test trained model on test dataloader
    '''
    
    # load saved model
    model.load_state_dict(torch.load(os.path.join(cache_dir, 'cifar.pt')))
    print('model loaded')

    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        print("Input:", inputs.shape)
        outputs = model(inputs)
        print('model out:', outputs.shape)
        if i == 1:
            break

    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    #dataiter = iter(testloader)
    #images, labels = dataiter.next()
    #print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# training set
training_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training')
trainset = torchvision.datasets.CIFAR10(root=training_folder, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
# test set
testset = torchvision.datasets.CIFAR10(root=training_folder, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
# cifar classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# train model to device
train(cnn_model, device, trainloader, cache_dir=cache_dir)
test(cnn_model, device, testloader, cache_dir=cache_dir)