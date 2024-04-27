import os

import caml_core as core
import caml_networks as networks

SAM_PATH = "/home/path/projects/oasis/Networks/third_party/segment_anything"


def annotate_masks_for_dataset(dataset_path: str, start_index: int = 0):
    # Setup SAM
    settings = networks.premade.instance_segmentation.SegmentAnythingSettings.from_yaml(os.path.join(SAM_PATH, "hparams.yaml"))
    settings.request_user_prompts = True
    sam_model = networks.deployment.SegmentAnything(settings=settings)

    # Get observations
    folders = list(
        filter(
            lambda folder: os.path.isdir(os.path.join(dataset_path, folder)),
            sorted(os.listdir(dataset_path)),
        )
    )

    for index in range(start_index, len(folders)):
        observation_path = os.path.join(dataset_path, folders[index])

        while True:
            observation = core.io.load_observation(observation_path)

            # Annotate image
            prediction, annotated_observation = sam_model.forward(observation=observation)
            annotated_observation.image.annotation_groups = annotated_observation.depth_image.annotation_groups
            annotated_observation.image.show(show_annotations=True)

            if core.logger.ask_proceed(question="Mask Good?", ask_proceed=True):
                mask = prediction.masks[0].to_image()
                core.io.save_image(os.path.join(observation_path, "mask.png"), mask)
                core.logger.info(f"Annotated observation {index}: {folders[index]}")
                break


if __name__ == "__main__":
    annotate_masks_for_dataset(
        dataset_path="/home/mohith/Documents/repos/config_classifier/data/v2-1",
        start_index=0,
    )
