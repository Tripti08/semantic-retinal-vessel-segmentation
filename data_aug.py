"""Paired augmentations for a retinal image and its binary mask."""

import random
import numpy as np
from PIL import Image


def random_pair_augmentation(image: Image.Image, mask: Image.Image):
    """Apply the same geometric transform to image and mask."""
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        turns = random.randint(1, 3)
        image = image.rotate(90 * turns)
        mask = mask.rotate(90 * turns)
    return image, mask


def mask_to_array(mask: Image.Image) -> np.ndarray:
    values = np.asarray(mask.convert("L"), dtype=np.float32)
    return (values > 127).astype(np.float32)

