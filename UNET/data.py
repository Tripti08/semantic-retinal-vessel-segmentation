from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_aug import mask_to_array, random_pair_augmentation

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_EXTENSIONS = IMAGE_EXTENSIONS | {".gif"}


def _files(directory: Path, extensions) -> List[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions)


class DriveDataset(Dataset):
    """DRIVE image/mask dataset with deterministic resizing."""

    def __init__(self, image_dir: str | Path, mask_dir: str | Path, image_size: Tuple[int, int] = (512, 512), augment=False):
        self.image_paths = _files(Path(image_dir), IMAGE_EXTENSIONS)
        self.mask_paths = _files(Path(mask_dir), MASK_EXTENSIONS)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Found {len(self.image_paths)} images but {len(self.mask_paths)} masks")
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")
        if self.augment:
            image, mask = random_pair_augmentation(image, mask)
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_array = mask_to_array(mask)[None, ...]
        return torch.from_numpy(image_array), torch.from_numpy(mask_array)


class _IndexedDataset(Dataset):
    def __init__(self, base, indices, augment):
        self.base, self.indices, self.augment = base, list(indices), augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        original_index = self.indices[item]
        image = Image.open(self.base.image_paths[original_index]).convert("RGB")
        mask = Image.open(self.base.mask_paths[original_index]).convert("L")
        if self.augment:
            image, mask = random_pair_augmentation(image, mask)
        image = image.resize(self.base.image_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.base.image_size, Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(image_array), torch.from_numpy(mask_to_array(mask)[None, ...])


def make_drive_datasets(data_dir: str | Path, image_size=(512, 512), val_fraction=0.2):
    root = Path(data_dir) / "training"
    full = DriveDataset(root / "images", root / "1st_manual", image_size=image_size)
    indices = np.arange(len(full))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_fraction)))
    val_indices = sorted(indices[:val_count].tolist())
    train_indices = [i for i in indices.tolist() if i not in val_indices]
    return _IndexedDataset(full, train_indices, True), _IndexedDataset(full, val_indices, False)
