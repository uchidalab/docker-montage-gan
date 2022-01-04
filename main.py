import subprocess
import os

outdir = "output"
data = "data/global"
mirror = "true"
metrics = "none"
kimg = 5000
target = 0.7
augpipe = "bgcfnc"
net_snap = 5
# cfg = "aio"
cfg = "aio_local"

env = os.environ.copy()
# Debug
env["CUDA_LAUNCH_BLOCKING"] = "1"

subprocess.run(
    f"python montage_gan/train_aio.py --outdir {outdir} --data {data}"
    f" --mirror {mirror} --metrics {metrics} --kimg {kimg}"
    f" --target {target} --augpipe {augpipe} --net-snap {net_snap} --cfg {cfg}".split(" "), env=env)
