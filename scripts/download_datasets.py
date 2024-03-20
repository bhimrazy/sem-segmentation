#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess
import sys

from decouple import config

# Load environment variables
DATASET_ID = config("DATASET_ID")
if DATASET_ID is None:
    raise ValueError("DATASET_ID not found in .env file")

# Define dataset IDs
DATASET_IDS = {
    "sem-dataset": DATASET_ID,
}


def download_dataset(file_id, destination_folder):
    """
    Download a dataset from Google Drive using gdown
    Args:
        file_id (str): The ID of the file to download
        destination_folder (str): The folder to download the dataset to
    """
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Download the file using gdown
    print("Downloading the dataset...")
    subprocess.run(
        [
            "gdown",
            f"https://drive.google.com/uc?id={file_id}",
            "-O",
            f"{destination_folder}/dataset.zip",
        ]
    )

    # Unzip the dataset
    print("Unzipping the dataset...")
    subprocess.run(
        ["unzip", f"{destination_folder}/dataset.zip", "-d", destination_folder]
    )

    # Remove the downloaded zip file
    os.remove(f"{destination_folder}/dataset.zip")

    # Remove the __MACOSX folder
    if os.path.exists(f"{destination_folder}/__MACOSX"):
        shutil.rmtree(f"{destination_folder}/__MACOSX")
    print("Dataset downloaded, unzipped, and zip file removed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Download dataset from Google Drive")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="sem-dataset",
        help="Dataset to download. Available datasets are: otitis-media, oral-cancer, and genetics",  # noqa: E501
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="data",
        help="Folder to download the dataset to",
    )
    args = parser.parse_args()

    # Check if gdown is installed
    if not shutil.which("gdown"):
        print("gdown is not installed. Installing gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"])

    # Check if the dataset is valid
    if args.dataset not in DATASET_IDS:
        print(
            "Invalid dataset. Available datasets are: otitis-media, oral-cancer, and genetics"  # noqa: E501
        )
        sys.exit(1)

    # Download the dataset
    download_dataset(DATASET_IDS[args.dataset], args.folder)


if __name__ == "__main__":
    main()
