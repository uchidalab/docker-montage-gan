import random

import torch

from training.augment import AugmentPipe


def apply_augment(blchw: torch.Tensor, augment_pipe: AugmentPipe):
    """
    A wrapper of AugmentPipe for global GAN
    """
    assert blchw.ndim == 5
    seed = random.randint(0, 2147483647)
    i_layers_t_list = []
    for i_layers_t in blchw.transpose(0, 1):  # Transpose blchw to lbchw.
        # Here we ensure that the same augmentation is applied to all layers in each lchw
        # by using a fixed random seed during this for loop.
        torch.manual_seed(seed)
        i_layers_t = augment_pipe(i_layers_t)
        i_layers_t_list.append(i_layers_t)
    return torch.stack(i_layers_t_list).transpose(0, 1).contiguous()  # Transpose lbchw back to blchw.
