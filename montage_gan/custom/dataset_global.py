import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset, T_co
from torchvision.transforms import ToTensor


class DatasetGlobal(Dataset):
    def __init__(self,
                 path,  # Path to directory.
                 xflip=True,  # Artificially double the size of the dataset via x-flips
                 ):
        self._path = path
        self._dirs = sorted(os.listdir(path))  # Directories to each data.

        # Apply xflip.
        """
        if xflip: raw_idx = [0, 1, ... , N-1, 0, 1, ..., N-1]; is_xflip = [0, 0, ..., 0, 1, 1, ..., 1]
        else:     raw_idx = [0, 1, ... , N-1]; is_xflip = [0, 0, ..., 0]
        """
        self.raw_idx = np.arange(len(self._dirs), dtype=np.int64)
        self.is_xflip = np.zeros(self.raw_idx.size, dtype=np.uint8)
        if xflip:
            self.raw_idx = np.tile(self.raw_idx, 2)
            self.is_xflip = np.concatenate([self.is_xflip, np.ones_like(self.is_xflip)])

    def __len__(self):
        return len(self.raw_idx)

    def __getitem__(self, index) -> T_co:
        """
        :param index: index
        :return: Numpy.array [L,C,H,W]
        """
        data_dir = os.path.join(self._path, self._dirs[self.raw_idx[index]])
        files = sorted(os.listdir(data_dir))
        layers = []
        for f in files:
            filepath = os.path.join(data_dir, f)
            # Load with Pillow
            img_pil = Image.open(filepath)
            # Convert to tensor [0.,1.]
            img_tensor = ToTensor()(img_pil)
            layers.append(img_tensor)
        layers = torch.stack(layers)

        # Process xflip.
        if self.is_xflip[index]:
            # Flip last dim
            layers = torch.flip(layers, [-1])

        return layers
