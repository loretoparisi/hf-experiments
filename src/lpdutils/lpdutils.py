# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import numpy as np
import torch
from torch import Tensor, device

class LPDUtils(object):
    '''
        Common Utilities
    '''
    
    @classmethod
    def tolabels(self, predictions, int2label_dict, model_type, decimal_places = 4):
        '''
            convert numeric predictions to labeled predictions
        '''
        
        labels = {}
        
        # numpy ndarray
        if model_type=='sklearn' and isinstance(predictions, (np.ndarray, np.generic)):
            for (index, value) in enumerate(predictions):
                label = int2label_dict[index]
                labels[label] = round(value,decimal_places)
        
        # torch tensor
        elif model_type=='bert' and torch.is_tensor(predictions):
            for (index, value) in enumerate(predictions):
                label = int2label_dict[index]
                labels[label] = round(value.item(), decimal_places)
        
        # fasttext (tuple, ndarray)
        elif model_type=='fasttext' and isinstance(predictions, tuple) and isinstance(predictions[1], np.ndarray):
            # (('__label__SPAM', '__label__VERIFIED'), array([1.00000739e+00, 1.26670984e-05]))
            # predictions are ordered to max prob
            int2label_dict = predictions[0]
            for (index, value) in enumerate(predictions[1]):
                label = int2label_dict[index].replace('__label__', '')
                labels[label] = round(value, decimal_places)
        
        return labels

    @classmethod
    def has_gpu(self):
        '''
            detect gpu available
        '''
        has_gpu = torch.cuda.is_available()
        return has_gpu
    
    @classmethod
    def get_device(self):
        '''
            one of cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, msnpu, xla, vulkan device
        '''
        device = 'cuda' if self.has_gpu() else 'cpu'
        return device
    
    @classmethod
    def cosine_similarity(self, a: Tensor, b: Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.mm(a_norm, b_norm.transpose(0, 1))