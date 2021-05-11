# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import numpy as np
import natsort
from PIL import Image
import torch
import torchvision.transforms as T

# ResMLP
from res_mlp_pytorch.res_mlp import ResMLP
# MLP-Mixer
from mlp_mixer.mlp_mixer import MLPMixer
# Perceiver, General Perception with Iterative Attention
from perceiver import Perceiver

class LPCustomDataSet(torch.utils.data.Dataset):
    '''
        Naive Torch Image Dataset Loader
        with support for Image loading errors
        and Image resizing
    '''
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        try:
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
            return tensor_image
        except:
            pass
            return None

    @classmethod
    def collate_fn(self, batch):
        '''
            Collate filtering not None images
        '''
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    @classmethod
    def transform(self,img):
        '''
            Naive image resizer
        '''
        transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(img)

# Res MLP
res_model = ResMLP(
    image_size = 224,
    patch_size = 16,
    dim = 512,
    depth = 12,
    num_classes = 1000
)

parameters = filter(lambda p: p.requires_grad, res_model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('ResMLP trainable Parameters: %.3fM' % parameters)

# input must have 256 channels
img = torch.randn(1, 3, 224, 224)
pred = res_model(img) # (1, 1000)
print("ResMLP:",pred.shape)

# MLP Mixer
mixer_model = MLPMixer(in_channels=3, 
                image_size=224, 
                patch_size=16, 
                num_classes=1000,
                dim=512, 
                depth=8, 
                token_dim=256, 
                channel_dim=2048)

parameters = filter(lambda p: p.requires_grad, mixer_model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('MLPMixer trainable Parameters: %.3fM' % parameters)

img = torch.ones([1, 3, 224, 224])
pred = mixer_model(img)
print("MLPMixer:", pred.shape) # [B, in_channels, image_size, image_size]

# Perceiver
perceiver_model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 6,                   # depth of net
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,
    latent_dim_head = 64,
    num_classes = 1000,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
    fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
    self_per_cross_attn = 2      # number of self attention blocks per cross attention
)
parameters = filter(lambda p: p.requires_grad, perceiver_model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Perceiver trainable Parameters: %.3fM' % parameters)
img = torch.randn(1, 224, 224, 3) # 1 imagenet image, pixelized
perceiver_model(img) # (1, 1000)

batch_size = 2
my_dataset = LPCustomDataSet(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data'), transform=LPCustomDataSet.transform)
train_loader = torch.utils.data.DataLoader(my_dataset , batch_size=batch_size, shuffle=False, 
                               num_workers=4, drop_last=True, collate_fn=LPCustomDataSet.collate_fn)
for idx, img in enumerate(train_loader):
    print(idx, img.shape)
    pred = res_model(img)
    print("ResMLP pred:", pred.shape)
    pred = mixer_model(img)
    print("MLPMixer pred:", pred.shape)