import os
from typing import Tuple

import yaml
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data.dataset import Dataset


class CCDataset(Dataset):

    def __init__(self, path: str, type: str = "train") -> None:

        self._path = path

        with open(f"{path}/{type}.txt", "r") as f:
            samples = f.readlines()
        self._samples = [sample.strip() for sample in samples]  # Strip newline characters from each line

        if type == "train":
            self._transforms = torchvision.transforms.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(degrees=180),
                    T.GaussianBlur(kernel_size=5, sigma=[0.1, 4]),
                    T.RandomGrayscale(),
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    T.RandomPerspective(distortion_scale=0.25),
                    T.Resize(size=250, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
                    T.RandomCrop(224),
                ]
            )
        else:
            self._transforms = torchvision.transforms.Compose(
                [
                    T.Resize(size=232, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
                    T.CenterCrop(224),
                ]
            )

    def __len__(self) -> int:
        return len(self._samples)

    def compute_stats(self) -> None:
        bolt_stats = {0: 0, 1: 0, 2: 0}
        hinge_stats = {0: 0, 1: 0, 2: 0}

        for sample in self._samples:
            label_path = os.path.join(self._path, sample, "label.yaml")
            label = yaml.safe_load(open(label_path, "r"))

            bolt_stats[label["bolt"]] += 1
            hinge_stats[label["hinge"]] += 1

        print("Bolt Stats: \n", bolt_stats)
        print("Hinge Stats: \n", hinge_stats)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self._path, self._samples[index], "image.png")
        image_tensor = torchvision.io.read_image(path=img_path)
        image_tensor = self._transforms(image_tensor)

        label_path = os.path.join(self._path, self._samples[index], "label.yaml")
        label = yaml.safe_load(open(label_path, "r"))
        label_tensor = torch.tensor([0 if label["bolt"] == 1 else 1], dtype=torch.float32)

        return image_tensor, label_tensor
