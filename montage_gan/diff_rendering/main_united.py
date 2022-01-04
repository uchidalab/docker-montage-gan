import copy
import datetime
import os
import json
from pprint import pprint

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from custom.dataset_global import DatasetGlobal as GlobalDataset
from custom_utils.image_utils import random_position, alpha_composite, normalize_zero1, normalize_minus11, calc_psnr
from custom_utils.utils import timestamp, strfdelta

config = {
    "dataset_root": "../data/global",
    "img_resolution": 256,
    "img_channels": 4,
    "img_layers": 9,
    "device": "cuda",
    "batch_size": 64,
    "seed": 0,
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "log_term": 10,
    "amsgrad": True,  # use amsgrad for Tanh variant
    "loss_type": "l1",  # l1/mse
    "save_term": 100,
    "train_iter": 32000,  # total step to train
    "resume_pth_path": None,  # pth to resume from
    "resume_step": None,  # step to resume from
    # "renderer_type": "tanh"  # sigmoid/tanh/subpixel
    "renderer_type": "subpixel"  # sigmoid/tanh/subpixel
}

# Use seed if provided
if config["seed"] is not None:
    torch.manual_seed(config["seed"])

is_range_tanh = True  # whether the data range is [0,1] or [-1,1]
if config["renderer_type"] == "sigmoid":
    from diff_rendering.networks import Renderer as Renderer  # Sigmoid variant

    is_range_tanh = False
elif config["renderer_type"] == "tanh":
    from diff_rendering.networks import RendererTanh as Renderer  # Tanh variant
elif config["renderer_type"] == "subpixel":
    from diff_rendering.networks import RendererSubPixelConv as Renderer  # Sub-pixel conv variant
else:
    raise RuntimeError(f"Unknown renderer type {config['renderer_type']}")

writer = SummaryWriter(comment=config["renderer_type"])

dataset = GlobalDataset(path=config["dataset_root"], xflip=True)
sample = copy.copy(dataset[0]).to(config["device"])
sample = torch.unsqueeze(sample, dim=0)  # [0,1]
writer.add_image("sample/alpha_blending", torch.squeeze(alpha_composite(sample)))
if is_range_tanh:
    sample = normalize_minus11(sample)  # [-1,1]

dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

renderer = Renderer(img_resolution=config["img_resolution"], img_channels=config["img_channels"],
                    img_layers=config["img_layers"]).to(config["device"])

optimizer = Adam(renderer.parameters(), lr=config["lr"], betas=config["betas"], amsgrad=config["amsgrad"])

if config["resume_pth_path"]:
    saved = torch.load(config["resume_pth_path"])
    if isinstance(saved, dict):
        # New pickle format
        config["resume_step"] = saved["global_step"]
        renderer.load_state_dict(saved["renderer"])
        optimizer.load_state_dict(saved["optimizer"])
    else:
        # Old pickle format that contain only the model, deprecated!
        renderer.load_state_dict(saved)

if config["loss_type"] == "mse":
    criterion = torch.nn.MSELoss()
elif config["loss_type"] == "l1":
    criterion = torch.nn.L1Loss()
else:
    raise RuntimeError(f"Unknown loss type {config['loss_type']}")


def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    torch.save(state, file_name)


output_dir = f"{timestamp()}-output-{config['renderer_type']}"
os.makedirs(output_dir, exist_ok=True)

# Log the config
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=3)
print("Current config:")
pprint(config)

start_time = datetime.datetime.now()
step = 0
if config["resume_step"]:
    step = config["resume_step"]

break_flag = False
while not break_flag:
    for i, data in enumerate(tqdm(dataloader, total=len(dataloader))):
        x = data.to(config["device"])  # [0,1]
        x = random_position(x)
        if is_range_tanh:
            x1 = normalize_minus11(x)  # [-1,1]
            output = renderer(x1)  # [-1,1]
            output = normalize_zero1(output)  # [0,1]
        else:
            output = renderer(x)  # [0,1]

        target = alpha_composite(x.detach())  # [0,1]
        target = target.to(config["device"])

        loss = criterion(output, target)
        psnr = calc_psnr(output.detach(), target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("psnr", psnr, step)

        is_last = step == config["train_iter"]

        if step % config["save_term"] == 0 or is_last:
            save_checkpoint({'global_step': step,
                             'renderer': renderer.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            '{}/renderer{:06d}.pth.tar'.format(output_dir, step))

        if step % config["log_term"] == 0 or is_last:
            time_delta = datetime.datetime.now() - start_time
            time_str = strfdelta(time_delta, "%D days %H:%M:%S")
            print('Step: {}, Loss: {:.4f}, PSNR: {:.2f}db, Time:{}'.format(step, loss.item(), psnr, time_str))
            with torch.no_grad():
                renderer.eval()
                output_eval = torch.squeeze(renderer(sample))
                if is_range_tanh:
                    output_eval = normalize_zero1(output_eval)
                writer.add_image("sample/renderer", output_eval, step)
                renderer.train()

        if is_last:
            break_flag = True
            break

        step += 1

writer.close()
print("The training has completed.")
