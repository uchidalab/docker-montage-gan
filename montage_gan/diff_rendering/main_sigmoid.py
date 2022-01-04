import copy
import datetime
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diff_rendering.networks import Renderer
from custom.dataset_global import DatasetGlobal as GlobalDataset
from custom_utils.image_utils import random_position, alpha_composite
from custom_utils.utils import timestamp, strfdelta

config = {
    "img_resolution": 256,
    "img_channels": 4,
    "img_layers": 9,
    "device": "cuda",
    "batch_size": 64,  # NOTE: 11/15, changed batch_size to 64
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "log_term": 10,
    "save_term": 100,
    "train_iter": 32000,  # total step to train
    "resume_pth_path": None,  # pth to resume from
    "resume_step": None,  # step to resume from
    "renderer_type": "sigmoid"  # sigmoid/tanh
}

writer = SummaryWriter(comment=config["renderer_type"])

# NOTE: 11/15, changed xflip to True
dataset = GlobalDataset(path="../data/global", xflip=True)
sample = copy.copy(dataset[0]).to(config["device"])
sample = torch.unsqueeze(sample, dim=0)
writer.add_image("sample/alpha_blending", torch.squeeze(alpha_composite(sample)))
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

renderer = Renderer(img_resolution=config["img_resolution"], img_channels=config["img_channels"],
                    img_layers=config["img_layers"]).to(config["device"])

optimizer = Adam(renderer.parameters(), lr=config["lr"], betas=config["betas"])

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

loss = torch.nn.MSELoss()

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

        output = renderer(x)  # [0,1]
        target = alpha_composite(x.detach())  # [0,1]
        target = target.to(config["device"])
        renderer_loss = loss(output, target)

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
            writer.add_image("sample/renderer", torch.squeeze(renderer(sample)), step)
            renderer.train()

        if is_last:
            break_flag = True
            break

        step += 1

writer.close()
print("The training has completed.")
