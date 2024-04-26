import os

import yaml
import numpy as np
import caml_core as core


def view_dataset(dataset_path: str) -> None:
    folders = filter(
        lambda folder: os.path.isdir(os.path.join(dataset_path, folder)),
        sorted(os.listdir(dataset_path)),
    )

    for folder in folders:
        observation_path = os.path.join(dataset_path, folder)
        observation = core.io.load_observation(observation_path)
        label = yaml.safe_load(open(os.path.join(observation_path, "label.yaml")))
        print(folder, ":\t", label)
        observation.image.show()


def split_dataset(dataset_path: str, train_fraction: float = 0.7) -> None:
    folders = os.listdir(dataset_path)
    np.random.shuffle(folders)
    index = int(np.round(train_fraction * len(folders)))
    train_set = folders[:index]
    test_set = folders[index:]

    with open(os.path.join(dataset_path, "train.txt"), "w") as f:
        f.writelines(f"{folder}\n" for folder in train_set)

    with open(os.path.join(dataset_path, "val.txt"), "w") as f:
        f.writelines(f"{folder}\n" for folder in test_set)


if __name__ == "__main__":
    pass
