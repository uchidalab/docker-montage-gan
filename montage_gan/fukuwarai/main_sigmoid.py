import datetime

from torch.autograd import grad
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from custom.dataset_global import DatasetGlobal as GlobalDataset
from custom_utils.image_utils import *
from custom_utils.utils import strfdelta, timestamp
from diff_rendering.networks import Renderer
from fukuwarai.networks import STNv2b as STN
from training.networks import Discriminator  # SG2ada Discriminator

config = {
    "img_resolution": 256,
    "img_channels": 4,
    "img_layers": 9,
    "device": "cuda",
    "batch_size": 16,
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "r1_gamma": 10,
    "log_term": 100,
    "save_term": 1000,
    "train_iter": 26000,  # total step to train
    "renderer_pth_path": None,  # Edit here
    "renderer_type": "sigmoid",  # sigmoid/tanh
}

writer = SummaryWriter(comment=config["renderer_type"])

# Networks
D = Discriminator(img_resolution=config["img_resolution"],
                  img_channels=config["img_channels"]).to(config["device"])
stn = STN(img_resolution=config["img_resolution"], img_channels=config["img_channels"],
          img_layers=config["img_layers"]).to(config["device"])
renderer = Renderer(img_resolution=config["img_resolution"], img_channels=config["img_channels"],
                    img_layers=config["img_layers"]).to(config["device"])
saved = torch.load(config["renderer_pth_path"])
if isinstance(saved, dict):
    renderer.load_state_dict(saved["renderer"])
else:
    renderer.load_state_dict(saved)
renderer.eval()

optimizer_d = Adam(D.parameters(), lr=config["lr"], betas=config["betas"])
optimizer_stn = Adam(stn.parameters(), lr=config["lr"], betas=config["betas"])

dataset = GlobalDataset(path="../data/global", xflip=True)
# Separate to 3 mini-batch
dataloader = DataLoader(dataset, batch_size=config["batch_size"] * 3, shuffle=True, drop_last=True)
loss = torch.nn.MSELoss()


# Reference
# https://github.com/Yangyangii/GAN-Tutorial/blob/master/CelebA/R1GAN.ipynb

def r1loss(inputs, label=None):
    # non-saturating loss with R1 regularization
    l = -1 if label else 1
    return F.softplus(l * inputs).mean()


def theta_constrain_loss(theta):
    # Constrain the theta parameters' range
    upper_bound = convert_translate_to_2x3(torch.tensor([[1., 1.]] * config["img_layers"], device=config["device"]))
    lower_bound = convert_translate_to_2x3(torch.tensor([[-1., -1.]] * config["img_layers"], device=config["device"]))
    clamp_theta = torch.max(torch.min(theta, upper_bound), lower_bound)
    return torch.norm(theta - clamp_theta, 2)


def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    torch.save(state, file_name)


output_dir = f"{timestamp()}-output-{config['renderer_type']}"
os.makedirs(output_dir, exist_ok=True)
start_time = datetime.datetime.now()
step = 0
break_flag = False

while not break_flag:
    for i, data in enumerate(tqdm(dataloader, total=len(dataloader))):
        data1, data2, data3 = torch.split(data, config["batch_size"])
        stn.zero_grad()
        renderer.zero_grad()

        # Training Discriminator
        x = alpha_composite(data1).to(config["device"])
        x = normalize_minus11(x)
        x.requires_grad = True  # This is required for "grad(outputs=x_outputs.sum(), inputs=x, create_graph=True)[0]"
        x_outputs = D(x)
        d_real_loss = r1loss(x_outputs, True)

        grad_real = grad(outputs=x_outputs.sum(), inputs=x, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty = 0.5 * config["r1_gamma"] * grad_penalty
        D_x_loss = d_real_loss + grad_penalty

        pseudo_fake = generate_pseudo_fake(data2).to(config["device"])
        x_fake, _ = stn(pseudo_fake)
        rendered = renderer(x_fake)
        z_outputs = D(normalize_minus11(rendered.detach()))
        D_z_loss = r1loss(z_outputs, False)
        D_loss = D_x_loss + D_z_loss

        D.zero_grad()
        D_loss.backward()
        optimizer_d.step()

        renderer.zero_grad()
        # Training Generator
        pseudo_fake = generate_pseudo_fake(data3).to(config["device"])
        x_fake, theta = stn(pseudo_fake)

        rendered = renderer(x_fake)
        alpha_blended = alpha_composite(x_fake.detach())
        # Calculate renderer loss just for checking. We don't train renderer here.
        # The renderer loss should be very small if working properly.
        renderer_loss = loss(rendered, alpha_blended)

        z_outputs = D(normalize_minus11(rendered))
        # grid = make_grid(rendered, padding=0)
        # writer.add_image("G/rendered", grid, step)
        G_r1_loss = r1loss(z_outputs, True)
        # Remove constrain for STNv2
        G_constrain_loss = theta_constrain_loss(theta)
        G_loss = G_r1_loss + G_constrain_loss
        # G_loss = G_r1_loss

        stn.zero_grad()
        G_loss.backward()
        optimizer_stn.step()

        # Logs
        writer.add_scalar("Renderer/loss", renderer_loss.item(), step)
        writer.add_scalar("G/loss", G_loss.item(), step)
        # Remove constrain for STNv2
        writer.add_scalar("G/loss/r1", G_r1_loss.item(), step)
        writer.add_scalar("G/loss/constrain", G_constrain_loss.item(), step)
        writer.add_scalar("D/loss", D_loss.item(), step)
        # Score
        writer.add_scalar("Score/fake", z_outputs.mean().item(), step)
        writer.add_scalar("Score/real", x_outputs.mean().item(), step)

        is_last = step == config["train_iter"]

        if step % config["save_term"] == 0 or is_last:
            save_checkpoint({'global_step': step,
                             'D': D.state_dict(),
                             'stn': stn.state_dict(),
                             'optimizer_d': optimizer_d.state_dict(),
                             'optimizer_stn': optimizer_stn.state_dict()},
                            '{}/r1gan{:06d}.pth.tar'.format(output_dir, step))

        if step % config["log_term"] == 0 or is_last:
            d_loss_str = "%.4f" % D_loss.item()
            g_loss_str = "%.4f" % G_loss.item()
            grad_penalty_str = "%.4f" % grad_penalty.item()
            time_delta = datetime.datetime.now() - start_time
            time_str = strfdelta(time_delta, "%D days %H:%M:%S")
            log_str = f"Step: {step}, D Loss: {d_loss_str}," \
                      f" G Loss: {g_loss_str}, gp: {grad_penalty_str}, Time:{time_str}"
            print(log_str)
            # Generate samples
            stn.eval()
            x_fake, theta = stn(pseudo_fake)
            rendered = renderer(x_fake)
            grid = make_grid(rendered, padding=0)
            writer.add_image("G/rendered", grid, step)
            stn.train()

        if is_last:
            break_flag = True
            break

        step += 1
