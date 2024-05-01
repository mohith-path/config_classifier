import os
from typing import List

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


def print_stats(dataset_path: str, folders: List[str]) -> None:
    bolt_stats = {0: 0, 1: 0, 2: 0}
    hinge_stats = {0: 0, 1: 0, 2: 0}

    for folder in folders:
        label_path = os.path.join(dataset_path, folder, "label.yaml")
        label = yaml.safe_load(open(label_path, "r"))

        bolt_stats[label["bolt"]] += 1
        hinge_stats[label["hinge"]] += 1

    print("Bolt Stats: \n", bolt_stats)
    print("Hinge Stats: \n", hinge_stats)


def split_dataset(dataset_path: str, train_fraction: float = 0.7) -> None:
    folders = list(
        filter(
            lambda folder: os.path.isdir(os.path.join(dataset_path, folder)),
            sorted(os.listdir(dataset_path)),
        )
    )

    val_only_folders = list(filter(lambda folder: "val" in folder, folders))
    unassigned_folders = list(filter(lambda folder: "val" not in folders, folders))
    print(f"Found {len(val_only_folders)} validation only folders.")

    train_count = int(train_fraction * len(folders))
    if train_count > len(unassigned_folders):
        print("WARNING! The specified train-val split is not respected due to a high number of val-only samples.")

    np.random.shuffle(unassigned_folders)

    train_set = unassigned_folders[:train_count]
    val_set = unassigned_folders[train_count:] + val_only_folders

    with open(os.path.join(dataset_path, "train.txt"), "w") as f:
        f.writelines(f"{folder}\n" for folder in train_set)

    with open(os.path.join(dataset_path, "val.txt"), "w") as f:
        f.writelines(f"{folder}\n" for folder in val_set)

    print("Dataset partition complete...")
    print("Train Set Stats:")
    print_stats(dataset_path, train_set)

    print("Val Set Stats:")
    print_stats(dataset_path, val_set)


if __name__ == "__main__":
    split_dataset("data/v3", train_fraction=0.75)
