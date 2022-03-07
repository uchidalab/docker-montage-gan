import subprocess
import os

"""
NOTE:
There are lots of configuration in the following scripts as well, make sure you check them:
training_loop_aio.py, dataset_aio.py
"""

"""
Arguments for SG2ada
"""
outdir = "output"
data = "data/global"
mirror = "true"
metrics = "none"
kimg = 5000
target = 0.6  # SG2-ada default is 0.6
augpipe = "bgcfnc"
net_snap = 5
cfg = "aio"

env = os.environ.copy()
# CUDA Debugging
# env["CUDA_LAUNCH_BLOCKING"] = "1"

# For step 1 (no resume)
# subprocess.run(
#     f"python montage_gan/train_aio.py --outdir {outdir} --data {data}"
#     f" --mirror {mirror} --metrics {metrics} --kimg {kimg}"
#     f" --target {target} --augpipe {augpipe} --net-snap {net_snap} --cfg {cfg}".split(" "), env=env)

# For step 2 (with resume)
resume = "pretrained/local_gans/network-snapshot-002500.pth"
subprocess.run(
    f"python montage_gan/train_aio.py --outdir {outdir} --data {data}"
    f" --mirror {mirror} --metrics {metrics} --kimg {kimg} --resume {resume}"
    f" --target {target} --augpipe {augpipe} --net-snap {net_snap} --cfg {cfg}".split(" "), env=env)

# For interpolation
resume = "output/00009-global-mirror-aio-kimg5000-target0.6-bgcfnc-resumecustom/network-snapshot-000100.pth"
# subprocess.run(
#     f"python montage_gan/train_aio.py --outdir {outdir} --data {data}"
#     f" --mirror {mirror} --metrics {metrics} --kimg {kimg} --resume {resume}"
#     f" --target {target} --augpipe {augpipe} --net-snap {net_snap} --cfg {cfg}".split(" "), env=env)

# For metrics evaluation
metrics = "fid50k_full,is50k"
resume = "output/00009-global-mirror-aio-kimg5000-target0.6-bgcfnc-resumecustom/network-snapshot-000100.pth"
# subprocess.run(
#     f"python montage_gan/train_aio.py --outdir {outdir} --data {data}"
#     f" --mirror {mirror} --metrics {metrics} --kimg {kimg} --resume {resume}"
#     f" --target {target} --augpipe {augpipe} --net-snap {net_snap} --cfg {cfg}".split(" "), env=env)
