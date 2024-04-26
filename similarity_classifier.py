import os.path
from typing import Dict

import cv2
import tqdm
import yaml
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
from featup.util import norm


class SimilarityClassifier:

    def __init__(self, dataset_path: str, device: str = "cpu") -> None:

        self._dataset_path = dataset_path
        self.device = device

        with open(f"{self._dataset_path}/train.txt", "r") as f:
            samples = f.readlines()
        self._samples = [sample.strip() for sample in samples]  # Strip newline characters from each line

        self._transforms = torchvision.transforms.Compose(
            [
                T.Resize(size=232, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
                T.CenterCrop(224),
                T.ToDtype(dtype=torch.float32, scale=True),
                norm,
            ]
        )

        self.feature_extractor = torch.hub.load("mhamilton723/FeatUp", "dino16", use_norm=True).to(self.device)
        self.feature_extractor.eval()
        self._create_lookup_database()

    @torch.no_grad()
    def _create_lookup_database(self) -> None:
        self._embeddings = []
        self._labels = []
        self._image_id = []

        for index in tqdm.tqdm(range(len(self._samples))):
            img_path = os.path.join(self._dataset_path, self._samples[index], "image.png")
            image_tensor = torchvision.io.read_image(path=img_path)
            image_tensor = self._transforms(image_tensor).to(self.device)

            label_path = os.path.join(self._dataset_path, self._samples[index], "label.yaml")
            label = yaml.safe_load(open(label_path, "r"))

            embedding = self.feature_extractor(image_tensor.unsqueeze(dim=0)).squeeze(dim=0)

            self._embeddings.append(embedding)
            self._labels.append(label)

        self._embeddings = torch.stack(self._embeddings, dim=0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = self._transforms(x).to(self.device).unsqueeze(dim=0)
        query = self.feature_extractor(x)[0]

        logits = torch.einsum("cwh,rcwh->r", [query, self._embeddings])
        # adopt the max similarity from different templates for each object
        index = logits.argmax(-1).detach().cpu().item()  # (M, R)

        return self._labels[index], index

    @staticmethod
    def process_label(label: Dict) -> int:
        return label["bolt"]

    def validate(self, visualize: bool = False) -> None:
        with open(f"data/v2-1/val.txt", "r") as f:
            samples = f.readlines()
        samples = [sample.strip() for sample in samples]  # Strip newline characters from each line

        actual_labels = []
        pred_labels = []

        for index in range(len(samples)):
            candidate_path = os.path.join(f"data/v2-1/{samples[index]}", "image.png")
            candidate_image_tensor = torchvision.io.read_image(path=candidate_path)

            candidate_label_path = os.path.join(f"data/v2-1/{samples[index]}", "label.yaml")
            candidate_label = yaml.safe_load(open(candidate_label_path, "r"))

            pred_label, anchor_index = classifier.forward(candidate_image_tensor)

            actual_labels.append(self.process_label(candidate_label))
            pred_labels.append(self.process_label(pred_label))

            # Visualize predictions on validation data
            if visualize and actual_labels[-1] != pred_labels[-1]:
                candidate_image = cv2.imread(candidate_path, flags=cv2.IMREAD_UNCHANGED)
                anchor_img_path = os.path.join(f"{self._dataset_path}/{self._samples[anchor_index]}", "image.png")
                anchor_image = cv2.imread(anchor_img_path, flags=cv2.IMREAD_UNCHANGED)

                print(f"Ground-Truth: {actual_labels[-1]} \t Predicted: {pred_labels[-1]}")
                stacked_image = cv2.hconcat([candidate_image, anchor_image])
                cv2.imshow("Best Match", stacked_image)
                cv2.waitKey(0)

        # Compute accuracy
        accuracy = np.mean(np.array(actual_labels) == np.array(pred_labels))
        print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    DATASET_PATH = "data/v2"

    classifier = SimilarityClassifier(dataset_path=DATASET_PATH, device="cuda")
    classifier.validate(visualize=True)
