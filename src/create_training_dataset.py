import os
from pathlib import Path
import shutil
import random

import pandas as pd
import argparse


def sample_image_paths(
    data_csv_filename: Path, benign_or_malignant: str, n_samples: int
) -> list[str]:
    if benign_or_malignant not in ["benign", "malignant"]:
        raise ValueError("Images to sample must be either 'benign' or 'malignant'")

    data = pd.read_csv(data_csv_filename)
    data = data[data["benign_malignant"] == benign_or_malignant]
    n_samples = min(n_samples, len(data))
    samples = random.sample(data["image_name"].tolist(), n_samples)

    return [f"{image}.jpg" for image in samples]


def create_subset_directory(
    source_dir: Path, target_dir: Path, n_samples: int
):
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Get a list of all files in the source directory
    all_files = os.listdir(source_dir)

    # Copy selected images to the target directory
    for image in all_files:
        src_path = os.path.join(source_dir, image)
        dest_path = os.path.join(target_dir, image)
        shutil.copyfile(src_path, dest_path)
        print(f"Copied '{image}' to '{target_dir}'")

    print(f"Subset of {n_samples} images created in '{target_dir}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a subset of images from a source directory"
    )
    parser.add_argument(
        "benign_or_malignant",
        type=str,
        choices=["benign", "malignant"],
        help="Whether to sample benign or malignant source images",
    )
    parser.add_argument(
        "--source_csv",
        type=str,
        help="Path to the CSV containing the metadata of the source data",
    )
    parser.add_argument("--source_dir", type=str, help="Path to the source directory")
    parser.add_argument("--target_dir", type=str, help="Path to the target directory")
    parser.add_argument("--subset_size", type=int, help="Number of images to copy")
    return parser.parse_args()


def main():
    args = parse_args()
    
    create_subset_directory(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        n_samples=args.subset_size,
    )


if __name__ == "__main__":
    main()
