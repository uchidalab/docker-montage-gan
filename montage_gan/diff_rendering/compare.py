import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from diff_rendering.networks import RendererTanh as Renderer  # Tanh variant
from custom.dataset_global import DatasetGlobal as GlobalDataset
from custom_utils.image_utils import alpha_composite, normalize_zero1, normalize_minus11

config = {
    "img_resolution": 256,
    "img_channels": 4,
    "img_layers": 9,
    "device": "cuda",
    "batch_size": 64,
    # "renderer_pth_path_1": None,
    "renderer_pth_path_1": "211120-1956-output-tanh/renderer015000.pth.tar",
    "renderer_type_1": "tanh",  # sigmoid/tanh
    # "renderer_pth_path_2": None,
    "renderer_pth_path_2": "211203-2337-output-tanh/renderer015000.pth.tar",
    "renderer_type_2": "tanh",  # sigmoid/tanh
}

dataset = GlobalDataset(path="../data/global", xflip=False)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True)


def calc_psnr(x, y, data_range=1):
    mse = torch.nn.MSELoss()(x, y)
    return 10 * math.log10(data_range ** 2 / mse.item())


psnr_list_1 = []
psnr_list_2 = []

for j, (renderer_pth_path, renderer_type, psnr_list) in enumerate(
        zip([config["renderer_pth_path_1"], config["renderer_pth_path_2"]],
            [config["renderer_type_1"], config["renderer_type_2"]],
            [psnr_list_1, psnr_list_2])):
    renderer = Renderer(img_resolution=config["img_resolution"], img_channels=config["img_channels"],
                        img_layers=config["img_layers"]).to(config["device"])
    saved = torch.load(renderer_pth_path)
    if isinstance(saved, dict):
        renderer.load_state_dict(saved["renderer"])
    else:
        renderer.load_state_dict(saved)
    renderer.eval()

    for i, data in enumerate(tqdm(dataloader, total=len(dataloader))):
        x = data.to(config["device"])  # [0,1]
        if renderer_type == "tanh":
            x1 = normalize_minus11(x)  # [-1,1]
            output = renderer(x1)  # [-1,1]
            output = normalize_zero1(output)  # [0,1]
        else:
            output = renderer(x)  # [0,1]

        target = alpha_composite(x.detach())  # [0,1]

        psnr_list.append(calc_psnr(output, target))

    print(f"renderer_{j + 1} Avg. PSNR: {np.mean(psnr_list)}")
