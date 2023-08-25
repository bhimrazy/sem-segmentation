import glob
import logging

import yaml


def load_config(path: str) -> dict:
    """Load yaml configuration file from path."""
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError as e:
        logging.error(f"Config file not found at {path}")
        raise e


def load_data(dataset_path: str):
    """
    Load data from the dataset path

    Args:
        dataset_path (str): Path to the dataset folder.

    Returns:
        Tuple[List[str], List[str]]: images, masks
    """

    images_pattern = f"{dataset_path}/images/*"
    masks_pattern = f"{dataset_path}/masks/*"

    images = sorted(glob.glob(images_pattern))
    masks = sorted(glob.glob(masks_pattern))

    return images, masks