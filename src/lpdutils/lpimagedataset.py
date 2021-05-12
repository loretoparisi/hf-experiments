# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import numpy as np
import natsort
from PIL import Image
import torch
import torchvision.transforms as T

class LPImageDataSet(torch.utils.data.Dataset):
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