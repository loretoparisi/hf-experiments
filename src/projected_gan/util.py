# -*- coding: utf-8 -*-
# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2022 Loreto Parisi (loretoparisi at gmail dot com)

import os
import random

import torch
import torchvision
import matplotlib
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def torch_tensors_to_pil_list(input_images):
    out_images = []
    for in_image in input_images:
        in_image = in_image.cpu().detach()
        out_image = torchvision.transforms.functional.to_pil_image(in_image).convert('RGB')
        out_images.append(out_image)
    return out_images
def pil_list_to_torch_tensors(pil_images):
    result = []
    for pil_image in pil_images:
        image = np.array(pil_image, dtype=np.uint8)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).unsqueeze(0)
        result.append(image)
    return torch.cat(result, dim=0)
def show_images(pil_images, nrow=4, size=14):
    """
    :param pil_images: list of images in PIL
    :param nrow: number of rows
    :param size: size of the images
    """
    
    matplotlib.use('TKAgg',force=True)
    import matplotlib.pyplot as plt

    pil_images = [pil_image.convert('RGB') for pil_image in pil_images]
    imgs = torchvision.utils.make_grid(pil_list_to_torch_tensors(pil_images), nrow=nrow)
    if not isinstance(imgs, list):
        imgs = [imgs.cpu()]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(size, size))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fix.show()
    plt.show()
