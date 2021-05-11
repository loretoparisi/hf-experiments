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

# MLP models: ResMLP, MLPMixer
from res_mlp_pytorch.res_mlp_pytorch import ResMLP
from mlp_mixer.mlp_mixer import MLPMixer

def imshow(img,images_path):
    '''
        Each CIFAR-10 image is a relatively small 32 x 32 pixels in size. 
        The images are in color so each pixel has three values for the red, green, and blue channel values. 
        Therefore, each image has a total of 32 * 32 * 3 = 3072 values.
    '''
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    matplotlib.use('agg')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(images_path, bbox_inches='tight', dpi=300)
    plt.close()

# Image normalize
transformNormalize = T.Compose(
    [T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Image resize 256
transformResize = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

class CNNish(nn.Module):
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

def predict(model, device, inputs, classes, cache_dir=''):
    '''
        test trained model on test dataloader
    '''
    
    # load saved model
    model.load_state_dict(torch.load(os.path.join(cache_dir, 'cifar.pt')))
    print('model loaded')

    inputs  = inputs.to(device)
    print("Input:", inputs.shape)
    outputs = model(inputs)
    print('model out:', outputs.shape)

    ########################################################################
    # The outputs are energies for the 10 classes.
    # The higher the energy for a class, the more the network
    # thinks that the image is of the particular class.
    # So, let's get the index of the highest energy:
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                            for j in range(4)))

def test(model, device, dataloader, classes, cache_dir=''):
    '''
        test trained model on test dataloader and 
        return accuracy by class
    '''
    
    # load saved model
    model.load_state_dict(torch.load(os.path.join(cache_dir, 'cifar.pt')))
    print('model loaded')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)    
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
                                                    accuracy))

# cache dir
cache_dir = os.getenv("cache_dir", "../../models")
# choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 4
num_workers = 2
training_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training')

'''
    There are a total of 60,000 CIFAR-10 images divided into 6,000 each of 10 (hence the “10” in “CIFAR-10”) different objects: 
    ‘plane’, ‘car’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. 
    There is also a CIFAR-100 dataset that has 100 different items.
'''
# training set
trainset = torchvision.datasets.CIFAR10(root=training_folder, train=True,
                                        download=True, transform=transformNormalize)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
# test set
testset = torchvision.datasets.CIFAR10(root=training_folder, train=False,
                                       download=True, transform=transformNormalize)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
# cifar classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# init models

# CNN like
cnn_model = CNNish()
cnn_model.to(torch.device(device))

# ResMLP
res_model = ResMLP(
    image_size = 32, # CIFAR10 image: 32 x 32 x 3 = 3072
    patch_size = 16,
    dim = 512,
    depth = 12,
    num_classes = len(classes)
)
res_model.to(torch.device(device))

# MLP-Mixer
mixer_model = MLPMixer(in_channels=3, 
                image_size=32, 
                patch_size=16, 
                num_classes=len(classes),
                dim=512, 
                depth=8, 
                token_dim=256, 
                channel_dim=2048)

mixer_model.to(torch.device(device))

# train model to device: cnn_model, mixer_model, res_model
model = res_model
train(model, device, trainloader, cache_dir=cache_dir)

# test trained model on testset
test(model, device, testloader, classes, cache_dir=cache_dir)

# predict trained model on random images of batch_size
dataiter = iter(testloader)
images, labels = dataiter.next()

# show images
images_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images.png')
imshow(torchvision.utils.make_grid(images), images_path)

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
predict(model, device, images, classes, cache_dir=cache_dir)