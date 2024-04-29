import os
from typing import List, Tuple

import yaml
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data.dataset import Dataset


class CCDataset(Dataset):

    BACKGROUND_IMAGES_PATH = "data/background"

    def __init__(self, path: str, type: str = "train", use_background_augmentation: bool = True) -> None:

        self._path = path
        self.type = type
        self.use_background_augmentation = use_background_augmentation

        with open(f"{path}/{type}.txt", "r") as f:
            samples = f.readlines()
        self._samples = [sample.strip() for sample in samples]  # Strip newline characters from each line

        if use_background_augmentation and type == "train":
            self._bg_images = list(
                filter(
                    lambda folder: os.path.isdir(os.path.join(self.BACKGROUND_IMAGES_PATH, folder)),
                    sorted(os.listdir(self.BACKGROUND_IMAGES_PATH)),
                )
            )

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self._path, self._samples[index], "image.png")
        image_tensor = torchvision.io.read_image(path=img_path)

        if self.type == "train" and self.use_background_augmentation and np.random.rand() > 0.2:

            def morph(x: torch.Tensor, scale: float, rotation: float, center: List = None) -> torch.Tensor:
                h, w = x.shape[-2:]
                x = T.functional.rotate(x, angle=rotation, center=center)
                x = T.functional.resize(x, size=(int(scale * h), int(scale * w)), antialias=True)
                x = T.functional.center_crop(x, output_size=(h, w))
                return x

            # Load a random background image
            background = np.random.choice(self._bg_images)
            bg_image = torchvision.io.read_image(path=os.path.join(self.BACKGROUND_IMAGES_PATH, background, "image.png"))

            # Load mask of the current sample
            mask_path = os.path.join(self._path, self._samples[index], "mask.png")
            mask = torchvision.io.read_image(mask_path)
            mask = (mask / 255.0).to(torch.uint8)

            # Instance center
            row_suppress = mask.amax(dim=-2)[0].numpy()
            col_suppress = mask.amax(dim=-1)[0].numpy()
            r_min = np.argmax(row_suppress)
            c_min = np.argmax(col_suppress)
            r_max = (row_suppress.shape[0] - 1) - np.argmax(row_suppress[::-1])
            c_max = (col_suppress.shape[0] - 1) - np.argmax(col_suppress[::-1])
            r_center = (r_min + r_max) // 2
            c_center = (c_min + c_max) // 2

            # Apply a random transformation
            scale = np.random.uniform(0.75, 1.25)
            rotation = np.random.uniform(0, 360)
            image_tensor = morph(image_tensor, scale=scale, rotation=rotation, center=(r_center, c_center))
            mask = morph(mask, scale=scale, rotation=rotation, center=(r_center, c_center))

            # Compose a new image
            image_tensor = image_tensor * mask + (1 - mask) * bg_image

        image_tensor = self._transforms(image_tensor)

        label_path = os.path.join(self._path, self._samples[index], "label.yaml")
        label = yaml.safe_load(open(label_path, "r"))
        label_tensor = torch.tensor(
            [
                0 if label["bolt"] == 1 else 1,
                0 if label["hinge"] == 1 else 1,
            ],
            dtype=torch.float32,
        )

        return image_tensor, label_tensor
