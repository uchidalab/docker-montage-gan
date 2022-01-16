from pprint import pp

import torch

snapshot_path = "../../output/00000-global-mirror-aio_local-kimg5000-target0.7-bgcfnc/network-snapshot-000000.pth"
resume_data = torch.load(snapshot_path)
pp(resume_data)
