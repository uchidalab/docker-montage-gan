import os
from pprint import pprint

import numpy as np
import torch
from PIL import Image
from numpy import array
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor

from custom_utils.calc_res import calc_res, calc_init_res

dataset_config = {
    "name": "global",
    "debug": False,
    # Result from dataset_stat, skip the calculation if defined
    "stat": {'layer_names': ['#1_hair_back',
                             '#2_body',
                             '#2_ear',
                             '#3_face',
                             '#4_eye',
                             '#4_mouth',
                             '#4_nose',
                             '#5_hair_front',
                             '#6_brow'],
             'layer_stats': {'#1_hair_back': {'avg_center': array([127.10861057, 114.34295499]),
                                              'base_height': 256,
                                              'base_width': 256,
                                              'max_center_diff': array([36.39138943, 81.65704501]),
                                              'required_height': 256,
                                              'required_width': 256},
                             '#2_body': {'avg_center': array([127.14334638, 193.29403131]),
                                         'base_height': 256,
                                         'base_width': 256,
                                         'max_center_diff': array([29.35665362, 65.29403131]),
                                         'required_height': 256,
                                         'required_width': 256},
                             '#2_ear': {'avg_center': array([128.04773869, 133.46080402]),
                                        'base_height': 160,
                                        'base_width': 224,
                                        'max_center_diff': array([96.04773869, 61.96080402]),
                                        'required_height': 142,
                                        'required_width': 222},
                             '#3_face': {'avg_center': array([126.91193738, 97.05919765]),
                                         'base_height': 256,
                                         'base_width': 256,
                                         'max_center_diff': array([31.41193738, 46.44080235]),
                                         'required_height': 231,
                                         'required_width': 256},
                             '#4_eye': {'avg_center': array([127.1037182, 108.14187867]),
                                        'base_height': 96,
                                        'base_width': 160,
                                        'max_center_diff': array([19.8962818, 19.85812133]),
                                        'required_height': 94,
                                        'required_width': 151},
                             '#4_mouth': {'avg_center': array([127.02886497, 155.49168297]),
                                          'base_height': 64,
                                          'base_width': 96,
                                          'max_center_diff': array([9.02886497, 14.00831703]),
                                          'required_height': 50,
                                          'required_width': 65},
                             '#4_nose': {'avg_center': array([126.41911765, 133.39852941]),
                                         'base_height': 64,
                                         'base_width': 32,
                                         'max_center_diff': array([21.08088235, 14.39852941]),
                                         'required_height': 50,
                                         'required_width': 31},
                             '#5_hair_front': {'avg_center': array([126.32272282, 103.12830558]),
                                               'base_height': 256,
                                               'base_width': 256,
                                               'max_center_diff': array([37.32272282, 42.62830558]),
                                               'required_height': 256,
                                               'required_width': 256},
                             '#6_brow': {'avg_center': array([126.54260529, 84.54750245]),
                                         'base_height': 64,
                                         'base_width': 160,
                                         'max_center_diff': array([43.95739471, 25.54750245]),
                                         'required_height': 41,
                                         'required_width': 133}},
             'shape': (1022, 9, 4, 256, 256)}
}


def dataset_stat(src, conv_base_index=2):
    from collections import defaultdict

    from tqdm import tqdm
    import cv2

    from custom_utils.utils import listdir_relative
    from custom_utils.calc_res import find_min_acceptable_res_combination, calc_acceptable_res_combination

    acceptable_res_combination = calc_acceptable_res_combination(conv_base_index)

    layers = defaultdict(list)
    stat = {"shape": None,
            "layer_names": [],
            "layer_stats": {}}
    img_shape = None
    dataset_len = len(os.listdir(src))
    for d in tqdm(listdir_relative(src), desc="Dataset statistic..."):
        for f in listdir_relative(d):
            layer_name = os.path.basename(f)[:-4]
            img = cv2.cvtColor(cv2.imread(f, -1), cv2.COLOR_BGRA2RGBA)
            r, g, b, a = cv2.split(img)
            coords = cv2.findNonZero(a)
            box = cv2.boundingRect(coords)
            # Ignore blank
            if box != (0, 0, 0, 0):
                layers[layer_name].append(box)
            # Store image shape
            if img_shape is None:
                img_shape = img.shape[2], img.shape[0], img.shape[1]

    # Convert to np.array
    for i in layers.items():
        key, value = i
        arr = np.array(value)
        # cv2.boundingRect -> x,y,w,h
        max_w, max_h = np.max(arr, axis=0)[2:]
        # x+w/2, y+h/2
        center = arr[:, :2] + arr[:, 2:] / 2
        avg_center = np.average(center, axis=0)
        # max(abs(center - avg_center))
        max_center_diff = np.max(np.abs(center - avg_center), axis=0)
        # base_res
        base_w, base_h = find_min_acceptable_res_combination((max_w, max_h), acceptable_res_combination,
                                                             conv_base_index)
        stat["layer_stats"][key] = {"required_width": max_w, "required_height": max_h,
                                    "base_width": base_w,
                                    "base_height": base_h,
                                    "avg_center": avg_center, "max_center_diff": max_center_diff}
    stat["layer_names"] = list(stat["layer_stats"].keys())
    # [N,L,C,H,W]
    stat["shape"] = dataset_len, len(stat["layer_names"]), *img_shape
    return stat


class DatasetAIO(Dataset):
    def __init__(self,
                 path,  # Path to directory.
                 xflip=True,  # Artificially double the size of the dataset via x-flips
                 conv_base_index=2,
                 **kwargs
                 ):
        self.name = dataset_config["name"]
        self.conv_base_index = conv_base_index

        if dataset_config["stat"] is not None:
            self._stat = dataset_config["stat"]
        else:
            self._stat = dataset_stat(path)

        if dataset_config["debug"]:
            print("Dataset statistic")
            pprint(self._stat)

        self.layer_names = self._stat["layer_names"]
        self.layer_stats = self._stat["layer_stats"]
        self._raw_shape = list(self._stat["shape"])
        assert len(self._raw_shape) == 5  # NLCHW

        self._path = path
        self._dirs = sorted(os.listdir(path))  # Directories to each data.

        # Apply xflip.
        """
        if xflip: raw_idx = [0, 1, ... , N-1, 0, 1, ..., N-1]; is_xflip = [0, 0, ..., 0, 1, 1, ..., 1]
        else:     raw_idx = [0, 1, ... , N-1]; is_xflip = [0, 0, ..., 0]
        """
        self.raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        self.is_xflip = np.zeros(self.raw_idx.size, dtype=np.uint8)
        if xflip:
            self.raw_idx = np.tile(self.raw_idx, 2)
            self.is_xflip = np.concatenate([self.is_xflip, np.ones_like(self.is_xflip)])

    @property
    def layer_shape(self):
        return self._raw_shape[1:]

    @property
    def image_shape(self):
        return self._raw_shape[2:]

    @property
    def num_layers(self):
        return self.layer_shape[0]

    @property
    def num_channels(self):
        return self.image_shape[0]

    @property
    def resolution(self):
        return calc_res(self.image_shape[1:])

    def __len__(self):
        return len(self.raw_idx)

    def __getitem__(self, index):
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

    # Added for train_e2e
    @property
    def res_log2(self):
        return int(np.ceil(np.log2(self.resolution)))

    @property
    def init_res(self):
        return calc_init_res(self.image_shape[1:], conv_base_index=self.conv_base_index)[0]
        # return [int(s * 2 ** (2 - self.res_log2)) for s in self.image_shape[1:]]

    # Added for AIO
    def res_log2_layer(self, layer_name):
        stat_layer = self.layer_stats[layer_name]
        layer_size = [stat_layer["base_height"], stat_layer["base_width"]]
        return [int(np.ceil(np.log2(s))) for s in layer_size]

    def init_res_layer(self, layer_name):
        stat_layer = self.layer_stats[layer_name]
        layer_size = [stat_layer["base_height"], stat_layer["base_width"]]
        return calc_init_res(layer_size, conv_base_index=self.conv_base_index)[0]
        # res_log2_layer = self.res_log2_layer(layer_name)
        # return [int(s * 2 ** (2 - res_log2)) for res_log2, s in zip(res_log2_layer, layer_size)]

    def resolution_layer(self, layer_name):
        stat_layer = self.layer_stats[layer_name]
        layer_size = [stat_layer["base_height"], stat_layer["base_width"]]
        return calc_res(layer_size)
        # return max(layer_size)

    def base_size_layer(self, layer_name):
        stat_layer = self.layer_stats[layer_name]
        layer_size = [stat_layer["base_height"], stat_layer["base_width"]]
        return layer_size


def test_dataset(root_path):
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid

    conv_base_index = 2
    dataset = DatasetAIO(root_path, conv_base_index=conv_base_index)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("Dataset length:", len(dataset))
    # Draw a sample batch from DataLoader
    data = next(iter(dataloader))
    print("Data shape:", data.shape)
    grid = make_grid(data.flatten(0, 1), nrow=9).numpy()
    grid = grid.transpose(1, 2, 0)
    Image.fromarray(grid).show()
    print("Layers:")
    for layer in dataset.layer_names:
        print(layer, "init_res", dataset.init_res_layer(layer), "res_log2", dataset.res_log2_layer(layer))


if __name__ == "__main__":
    test_dataset("../../data/global")
