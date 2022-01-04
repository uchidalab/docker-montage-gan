import copy
import os

import torch

from diff_rendering.networks import Renderer
from custom.dataset_global import DatasetGlobal as GlobalDataset
from custom_utils.image_utils import alpha_composite, show_image

config = {
    "img_resolution": 256,
    "img_channels": 4,
    "img_layers": 9,
    "device": "cuda",
    "renderer_pth_path": None,
    "renderer_type": "sigmoid",  # sigmoid/tanh
}

dataset = GlobalDataset(path="../data/global", xflip=False)
sample = copy.copy(dataset[0]).to(config["device"])
sample = torch.unsqueeze(sample, dim=0)
del dataset

if config["renderer_pth_path"] is None:
    dirs = list(filter(lambda d: os.path.isdir(d) and d.endswith(f"output-{config['renderer_type']}"), os.listdir()))
    root = sorted(dirs)[-1]
    pth_path_list = [os.path.join(root, p) for p in os.listdir(root)]
    pth_path_list = list(filter(lambda p: os.path.isfile(p), pth_path_list))
    pth_path = sorted(pth_path_list)[-1]
    print(f"Evaluate using {pth_path}")
else:
    pth_path = config["renderer_pth_path"]

renderer = Renderer(img_resolution=config["img_resolution"], img_channels=config["img_channels"],
                    img_layers=config["img_layers"]).to(config["device"])

saved = torch.load(config["resume_pth_path"])
if isinstance(saved, dict):
    renderer.load_state_dict(saved["renderer"])
else:
    renderer.load_state_dict(saved)

renderer.eval()
# print(renderer)

loss = torch.nn.MSELoss()
output = renderer(sample)
target = alpha_composite(sample)
target = target.to(config["device"])
renderer_loss = loss(output, target)
print(renderer_loss.item())

print("--- Output ---")
print("min {:.2f} max {:.2f} mean {:.2f}".format(output.min().item(), output.max().item(), output.mean().item()))
print(output)
print("--- Target ---")
print("min {:.2f} max {:.2f} mean {:.2f}".format(target.min().item(), target.max().item(), target.mean().item()))
print(target)
print("--- Loss ---")
print("{:.2f}".format(renderer_loss.item()))

show_image(output)
show_image(target)
