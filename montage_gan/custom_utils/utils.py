import os
import re
import io
import traceback
import unicodedata
import shutil
from datetime import datetime
from string import Template

from PIL import Image
import torch.nn as nn
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    # https://stackoverflow.com/questions/8906926/formatting-timedelta-objects
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def timestamp():
    now = datetime.now()
    return now.strftime("%y%m%d-%H%M")


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Convert characters that aren't alphanumerics,
    underscores, or hyphens to dash. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '-', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


class ShapeChecker(nn.Module):
    """
    A tiny module for checking the input shape of the NN layer
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"[{self.name}] Shape: {x.shape}")
        return x


def bytes_to_pil(bytes):
    return Image.open(io.BytesIO(bytes))


class TensorBoardLogReader:
    """
    Reference: https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
    """

    def __init__(self, log_path, size_guidance=None):
        if size_guidance is None:
            size_guidance = {
                "scalars": 0,  # 0 means load all data.
                "images": 100,  # Tensorboard default is 4. Be careful not to cause OOM.
            }
        self.event_acc = EventAccumulator(log_path, size_guidance)
        self.name = os.path.basename(log_path)
        self.event_acc.Reload()
        self.tags = self.event_acc.Tags()

    def extract(self, out_root=None):
        if out_root is None:
            out_root = self.name
        if os.path.isdir(out_root):
            print("Output dir already exist. Deleting old data.")
            shutil.rmtree(out_root)
        try:
            # Scalars
            for tag in self.tags["scalars"]:
                out_path = os.path.join(out_root, f"{tag}.csv")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                event_list = self.event_acc.Scalars(tag)
                values = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x: x.step, event_list))
                r = {"value": values, "step": step}
                r = pd.DataFrame(r)
                r.to_csv(out_path)
            # Images
            for tag in self.tags["images"]:
                out_dir = os.path.join(out_root, tag)
                os.makedirs(out_dir, exist_ok=True)
                event_list = self.event_acc.Images(tag)
                images = list(map(lambda x: bytes_to_pil(x.encoded_image_string), event_list))
                step = list(map(lambda x: x.step, event_list))
                for img, s in zip(images, step):
                    img.save(os.path.join(out_dir, "step_{:06d}.png".format(s)))
        # Dirty catch of DataLossError
        except Exception:
            print("Event file possibly corrupt")
            traceback.print_exc()


def listdir_relative(path):
    return sorted([os.path.join(path, i) for i in os.listdir(path)])


def filter_isdir(path_list):
    return list(filter(lambda x: os.path.isdir(x), path_list))


def walk_layer_dir(path):
    filepath = []
    drawing_order = []
    for root, d_names, f_names in os.walk(path):
        for f in f_names:
            # Only append if *.png file
            if f.endswith(".png"):
                filepath.append(os.path.join(root, f))
                # Keep drawing order for checking
                drawing_order.append(int(f[:4]))
    # Sort png files only with their basename
    return sorted(filepath, key=lambda x: os.path.basename(x)), drawing_order


if __name__ == "__main__":
    # reader = TensorBoardLogReader("../fukuwarai/runs/Nov21_22-50-55_uchida-lab-jefftanh/events.out.tfevents.1637502656.uchida-lab-jeff.1884827.0")
    # reader.extract()
    # reader = TensorBoardLogReader("../fukuwarai/old/runs/Nov10_12-52-41_uchida-lab-jeff/events.out.tfevents.1636516362.uchida-lab-jeff.1064061.0")
    # reader.extract()
    pass
