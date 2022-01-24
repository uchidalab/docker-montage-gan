import subprocess
import os

outdir = "output"
data = "data/global"
mirror = "true"
metrics = "none"
kimg = 5000
# target = 0.7  # SG2-ada default is 0.6
target = 0.6  # SG2-ada default is 0.6
augpipe = "bgcfnc"
net_snap = 5
cfg = "aio"
resume = "pretrained/local_gans/network-snapshot-002500.pth"

env = os.environ.copy()
# Debug
# env["CUDA_LAUNCH_BLOCKING"] = "1"

# subprocess.run(
#     f"python montage_gan/train_aio.py --outdir {outdir} --data {data}"
#     f" --mirror {mirror} --metrics {metrics} --kimg {kimg}"
#     f" --target {target} --augpipe {augpipe} --net-snap {net_snap} --cfg {cfg}".split(" "), env=env)

# With resume
# subprocess.run(
#     f"python montage_gan/train_aio.py --outdir {outdir} --data {data}"
#     f" --mirror {mirror} --metrics {metrics} --kimg {kimg} --resume {resume}"
#     f" --target {target} --augpipe {augpipe} --net-snap {net_snap} --cfg {cfg}".split(" "), env=env)

# 0120 aug=0.3
# subprocess.run(
#     f"python montage_gan/train_aio.py --outdir {outdir} --data {data}"
#     f" --mirror {mirror} --metrics {metrics} --kimg {kimg} --resume {resume}"
#     f" --aug fixed --p 0.3 --augpipe {augpipe} --net-snap {net_snap} --cfg {cfg}".split(" "), env=env)

# Metrics
metrics = "fid50k_full,is50k"
resume = "output/00009-global-mirror-aio-kimg5000-target0.6-bgcfnc-resumecustom/network-snapshot-000120.pth"
subprocess.run(
    f"python montage_gan/train_aio.py --outdir {outdir} --data {data}"
    f" --mirror {mirror} --metrics {metrics} --kimg {kimg} --resume {resume}"
    f" --target {target} --augpipe {augpipe} --net-snap {net_snap} --cfg {cfg}".split(" "), env=env)