from copy import deepcopy
import torchvision.transforms as transforms
import numpy as np

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_im(dset, idx):
    '''Get numpy and torch image from dataset
    '''
    im = dset[idx][0]
    im_np = deepcopy(im.numpy()).transpose((1, 2, 0))
    im_torch = normalize(im).unsqueeze_(0)
    return im_np, im_torch


def to_np(im_torch, unnormalize=False):
    '''convert im_torch back to unnormalized numpy im
    1 x 3 x 224 x 224 -> 224 x 224 x 3
    '''
    return deepcopy(im_torch.cpu().detach().numpy()[0]).transpose((1, 2, 0))

def unnormalize(im_np):
    means = np.array([0.485/0.229, 0.456/0.224, 0.406/0.255]).T
    stds = np.array([0.229, 0.224, 0.255]).T
    im_np +=  means
    im_np *=  stds
    return im_np
    
    