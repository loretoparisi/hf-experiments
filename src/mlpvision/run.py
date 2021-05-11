# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import numpy as np
import natsort
from PIL import Image
import torch
import torchvision.transforms as T
from res_mlp_pytorch.res_mlp_pytorch import ResMLP
from mlp_mixer.mlp_mixer import MLPMixer

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
print("MLPMixer:", pred.shape) # # [B, in_channels, image_size, image_size]

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