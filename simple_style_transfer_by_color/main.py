"""
source:
https://github.com/YanchaoYang/FDA
"""

import os
from pathlib import Path

import numpy as np
import albumentations as A
import cv2


def load_images(path):
    images = []
    valid_images = [".jpeg", ".jpg", ".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(os.path.join(path, f))
    return sorted(images)


def agument_images(
    source_images_paths: Path, target_images_paths: Path, output_images_paths: Path
) -> None:
    output_images_paths.mkdir(exist_ok=True, parents=True)
    aug = A.Compose(
        [A.FDA(load_images(source_images_paths.as_posix()), p=1, beta_limit=0.0001)]
    )
    for image_path in load_images(target_images_paths.as_posix()):
        image_name = Path(image_path).name
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = aug(image=image)["image"]
        new_image_path = output_images_paths.joinpath(image_name).as_posix()
        cv2.imwrite(new_image_path, cv2.hconcat([image, result]))


if __name__ == "__main__":
    source_images_paths = Path("J:/deepcloth/datasets/deep_fashion2/train/image")
    target_images_paths = Path("sims_cloths")
    output_images_paths = Path("sims_cloths_agumented")
    agument_images(
        source_images_paths=source_images_paths,
        target_images_paths=target_images_paths,
        output_images_paths=output_images_paths,
    )
