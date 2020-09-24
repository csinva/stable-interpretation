import numpy as np
import torch
import acd
from copy import deepcopy
import sys
from numpy.fft import *
from torch import nn
from captum.attr import *
from . import viz

def get_attributions(x_t: torch.Tensor, 
                     mt, 
                     class_num=1,
                     attr_methods = ['Gradient', 'IG', 'DeepLift', 'SHAP', 'CD', 'InputXGradient'],
                     device='cuda'):
    '''Returns attribution maps in a dictionary for different attribution methods

    Params
    ------
    mt: pytorch model
    class_num: int
        target class
    
    Returns
    -------
    results: dict
        Dictionary of attribution maps
    '''
    x_t = x_t.to(device)
    x_t.requires_grad = True
    mt = mt.to(device)
    mt.eval()

    results = {}
    attr_funcs_dict = {
        'Gradient': None,
        'IG': IntegratedGradients,
        'DeepLift': DeepLift,
        'SHAP': GradientShap,
        'CD': None,
        'InputXGradient': InputXGradient
    }
        
    for name in attr_methods:
        func = attr_funcs_dict[name]
        if name == 'CD':
            with torch.no_grad():
                sweep_dim = 1
                tiles = acd.tiling_2d.gen_tiles(x_t[0,0,...,0], fill=0, method='cd', sweep_dim=sweep_dim)
                if x_t.shape[-1] == 2: # check for imaginary representations
                    tiles = np.repeat(np.expand_dims(tiles, axis=-1), repeats=2, axis=3).squeeze()
                tiles = torch.Tensor(tiles).unsqueeze(1)
                attributions = acd.get_scores_2d(mt, method='cd', ims=tiles, im_torch=x_t)[..., class_num].T.reshape(-1, 28, 28).squeeze()
                # attributions = score_funcs.get_scores_2d(mt, method='cd', ims=tiles, im_torch=x_t)[..., class_num].T.reshape(-1,28,28)
        elif name == 'Gradient':
            mt(x_t)[0, class_num].backward() # calculate gradients
            attributions = viz.detach(x_t.grad)
        else:
            baseline = torch.zeros(x_t.shape).to(device)
            attributer = func(mt)
            if name in ['InputXGradient']:
                attributions = attributer.attribute(deepcopy(x_t), target=class_num)
            else:
                attributions = attributer.attribute(deepcopy(x_t), deepcopy(baseline), target=class_num)
            attributions = attributions.cpu().detach().numpy().squeeze()
            if x_t.shape[-1] == 2: # check for imaginary representations
                attributions = mag(attributions)
        results[name] = attributions
    return results    