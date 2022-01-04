import copy
import datetime
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diff_rendering.networks import RendererTanh as Renderer  # Tanh variant
from custom.dataset_global import DatasetGlobal as GlobalDataset
from custom_utils.image_utils import random_position, alpha_composite, normalize_zero1, normalize_minus11
from custom_utils.utils import timestamp, strfdelta

config = {
    "img_resolution": 256,
    "img_channels": 4,
    "img_layers": 9,
    "device": "cuda",
    "batch_size": 64,  # NOTE: 11/15, changed batch_size to 64
    "lr": 1e-3,
    # "lr": 1e-5,
    "betas": (0.9, 0.999),
    "log_term": 10,
    "save_term": 100,
    "train_iter": 32000,  # total step to train
    # "resume_pth_path": None,  # pth to resume from
    "resume_pth_path": "211203-2337-output-tanh/renderer015000.pth.tar",  # pth to resume from
    # "resume_step": None,  # step to resume from
    "resume_step": 15000,  # step to resume from
    "renderer_type": "tanh"  # sigmoid/tanh
}

writer = SummaryWriter(comment=config["renderer_type"]+"_loss_mae")

# NOTE: 11/15, changed xflip to True
dataset = GlobalDataset(path="../data/global", xflip=True)
sample = copy.copy(dataset[0]).to(config["device"])
sample = torch.unsqueeze(sample, dim=0)  # [0,1]
writer.add_image("sample/alpha_blending", torch.squeeze(alpha_composite(sample)))
sample = normalize_minus11(sample)  # [-1,1]

dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

renderer = Renderer(img_resolution=config["img_resolution"], img_channels=config["img_channels"],
                    img_layers=config["img_layers"]).to(config["device"])

# optimizer = Adam(renderer.parameters(), lr=config["lr"], betas=config["betas"])

optimizer = Adam(renderer.parameters(), lr=config["lr"], betas=config["betas"],
                 amsgrad=True)  # Use amsgrad for Tanh variant

# Default parameter of AdamW
# lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False
# optimizer = AdamW(renderer.parameters(), lr=config["lr"], betas=config["betas"])

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

# loss = torch.nn.MSELoss()
loss = torch.nn.L1Loss()

b, l, c, h, w = config["batch_size"], config["img_layers"], config["img_channels"], config["img_resolution"], \
                config["img_resolution"]


def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    torch.save(state, file_name)


output_dir = f"{timestamp()}-output-{config['renderer_type']}"
os.makedirs(output_dir, exist_ok=True)
start_time = datetime.datetime.now()
step = 0
if config["resume_step"]:
    step = config["resume_step"]
break_flag = False

while not break_flag:
    for i, data in enumerate(tqdm(dataloader, total=len(dataloader))):
        x = data.to(config["device"])  # [0,1]
        x = random_position(x)
        x1 = normalize_minus11(x)  # [-1,1]

        output = renderer(x1)  # [-1,1]
        target = alpha_composite(x.detach())  # [0,1]
        target = target.to(config["device"])
        # NOTE: 11/20, calculate loss with [0,1] for a better comparison with Sigmoid variant
        output = normalize_zero1(output)  # [0,1]
        renderer_loss = loss(output, target)
        # Previous formulation
        # target = normalize_minus11(target)  # [-1,1]
        # renderer_loss = loss(output, target)

        optimizer.zero_grad()
        renderer_loss.backward()
        optimizer.step()

        writer.add_scalar("loss", renderer_loss.item(), step)

        is_last = step == config["train_iter"]

        if step % config["save_term"] == 0 or is_last:
            save_checkpoint({'global_step': step,
                             'renderer': renderer.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            '{}/renderer{:06d}.pth.tar'.format(output_dir, step))
            # torch.save(renderer.state_dict(), '{}/renderer{:06d}.pth.tar'.format(output_dir, step))

        if step % config["log_term"] == 0 or is_last:
            time_delta = datetime.datetime.now() - start_time
            time_str = strfdelta(time_delta, "%D days %H:%M:%S")
            print('Step: {}, Loss: {:.4f}, Time:{}'.format(step, renderer_loss.item(), time_str))
            renderer.eval()
            writer.add_image("sample/renderer", normalize_zero1(torch.squeeze(renderer(sample))), step)
            renderer.train()

        if is_last:
            break_flag = True
            break

        step += 1

writer.close()
print("The training has completed.")
