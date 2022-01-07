from pprint import pp

import torch

snapshot_path = "../../pretrained/snapshot/network-snapshot-000200.pth"
resume_data = torch.load(snapshot_path)
pp(resume_data)
