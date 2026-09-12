"""Dataset utilities for model-ready R2-BDA patches."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2

def _to_hwc(image):
    """Return a TIFF/array as HWC without guessing from spatial dimensions."""
    if image.ndim == 2:
        return image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got shape {image.shape}")
    if image.shape[0] <= 16 and image.shape[-1] > 16:
        return np.moveaxis(image, 0, -1)
    return image


NORMALIZATION_MODES = (
    "global_percentile_1_99_v1",
    "per_channel_percentile_1_99_v1",
)


def _normalise_image(image, mode="global_percentile_1_99_v1"):
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode: {mode}")
    image = image.astype(np.float32, copy=False)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)
    if mode == "global_percentile_1_99_v1":
        low, high = np.nanpercentile(image[finite], (1, 99))
        if high <= low:
            high = low + 1.0
        return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)
    output = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        values = image[..., channel][finite[..., channel]]
        if not len(values):
            continue
        low, high = np.nanpercentile(values, (1, 99))
        output[..., channel] = np.clip(
            (image[..., channel] - low) / max(high - low, 1.0), 0.0, 1.0
        )
    return output


def _resolve_mask_path(base_dir, record):
    """Return the indexed mask path or infer it from the image path."""
    mask_path = record.get("mask_path")
    if pd.notna(mask_path) and str(mask_path).strip():
        path = str(mask_path)
    else:
        parts = list(Path(str(record["image_path"])).parts)
        try:
            parts[parts.index("images")] = "masks"
        except ValueError as exc:
            raise ValueError(
                f"Cannot infer mask_path from image_path: {record['image_path']}"
            ) from exc
        path = str(Path(*parts))
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


class BuildingInstanceDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        base_dir,
        split,
        sensors,
        transform=None,
        use_mask=False,
        mask_ablation=None,
        channel_indices=None,
        normalization="global_percentile_1_99_v1",
        building_ids=None,
        mask_mode=None,
    ):
        """Load one split of model-ready patches."""
        self.base_dir = base_dir
        self.transform = transform
        self.use_mask = use_mask
        self.mask_ablation = mask_ablation
        self.mask_mode = mask_mode
        self.channel_indices = (
            tuple(int(index) for index in channel_indices)
            if channel_indices is not None
            else None
        )
        if normalization not in NORMALIZATION_MODES:
            raise ValueError(f"Unknown normalization mode: {normalization}")
        self.normalization = normalization

        df = pd.read_parquet(metadata_path)

        if building_ids is not None:
            df = df[df['building_id'].astype(str).isin(set([str(b) for b in building_ids]))]
        else:
            df = df[df['split'] == split]

        if isinstance(sensors, str):
            sensors = [sensors]
        df = df[df['sensor'].isin(sensors)]

        self.records = df.to_dict('records')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        img_path = os.path.join(self.base_dir, record['image_path'])

        if img_path.endswith('.tif') or img_path.endswith('.tiff'):
            image = _to_hwc(tifffile.imread(img_path))
        else:
            image = np.array(Image.open(img_path).convert('RGB'))
        if self.channel_indices is not None:
            if not self.channel_indices:
                raise ValueError("channel_indices must not be empty.")
            if min(self.channel_indices) < 0 or max(self.channel_indices) >= image.shape[-1]:
                raise ValueError(
                    f"Channel indices {self.channel_indices} exceed image channel range "
                    f"[0, {image.shape[-1] - 1}]: {img_path}"
                )
            image = image[..., list(self.channel_indices)]
        image = _normalise_image(image, self.normalization)

        mask = None
        if self.use_mask:
            mask_path = _resolve_mask_path(self.base_dir, record)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Missing footprint mask for building_id={record['building_id']}: {mask_path}"
                )
            mask = tifffile.imread(mask_path)
            if mask.ndim == 3:
                mask = np.squeeze(mask)
            if mask.ndim != 2:
                raise ValueError(f"Footprint mask must be single-channel; got {mask.shape}: {mask_path}")
            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"Image and mask dimensions differ: image={image.shape[:2]}, mask={mask.shape}, "
                    f"building_id={record['building_id']}"
                )
            mask = (mask > 0).astype(np.float32)

            if self.mask_ablation == "FC":
                h, w = mask.shape
                mask = np.zeros((h, w), dtype=np.float32)
                cy, cx = h // 2, w // 2
                mask[cy-22:cy+23, cx-22:cx+23] = 1.0
        if self.transform:
            augmented = self.transform(image=image, mask=mask) if mask is not None else self.transform(image=image)
            image = augmented['image']
            if mask is not None:
                mask = augmented["mask"]
                if not torch.is_tensor(mask):
                    mask = torch.from_numpy(mask)
                mask = mask.float().unsqueeze(0) if mask.ndim == 2 else mask.float()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if mask is not None:
                mask = torch.from_numpy(mask).float().unsqueeze(0)

        if mask is not None and self.mask_mode == "matched_support_14x14":
            from models.matched_support import compute_fc_support_14x14, compute_fi_support_14x14
            if self.mask_ablation == "FC":
                supp = compute_fc_support_14x14()
            else:
                m_arr = mask.squeeze().cpu().numpy() if torch.is_tensor(mask) else np.squeeze(mask)
                supp = compute_fi_support_14x14((m_arr > 0.5).astype(np.float32))
            mask = torch.from_numpy(supp).float()

        label_4class = record.get('label_4class', -1)
        label_2class = record.get('label_2class', -1)

        if pd.notna(label_4class) and label_4class != -1:
            label_4class = int(label_4class) - 1
        else:
            label_4class = -1

        if pd.notna(label_2class) and label_2class != -1:
            label_2class = int(label_2class)
        else:
            label_2class = -1

        sample = {
            'image': image,
            'label_4class': torch.tensor(label_4class, dtype=torch.long),
            'label_2class': torch.tensor(label_2class, dtype=torch.long),
            'building_id': str(record['building_id']),
            'sensor': record['sensor']
        }
        if mask is not None:
            sample["mask"] = mask
        return sample

def get_dataloaders(metadata_path, base_dir, sensors, batch_size=32, num_workers=4, use_mask=False):
    """Return dataloaders for the requested splits."""
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.2),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        ToTensorV2()
    ])

    dataloaders = {}

    for split, transform in [('train', train_transform), ('val', val_transform), ('test', val_transform)]:
        ds = BuildingInstanceDataset(
            metadata_path=metadata_path,
            base_dir=base_dir,
            split=split,
            sensors=sensors,
            transform=transform,
            use_mask=use_mask,
        )
        if len(ds) > 0:
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True
            )
            dataloaders[split] = dl
            print(f"Created DataLoader for {split} with {len(ds)} samples.")

    return dataloaders

if __name__ == "__main__":
    base = os.environ.get("R2_BDA_DATA_DIR", "data")
    meta = os.path.join(base, "09_metadata", "model_ready_patch_index.parquet")

    if os.path.exists(meta):
        print("Testing PyTorch Dataset loading...")
        dls = get_dataloaders(meta, base, sensors=['turkey_wv_visual_rgb'], batch_size=4, num_workers=0)

        if 'train' in dls:
            batch = next(iter(dls['train']))
            print(f"\nBatch Image shape: {batch['image'].shape}")
            print(f"Batch label_4class: {batch['label_4class']}")
            print(f"Batch label_2class: {batch['label_2class']}")
            print(f"Batch sensor: {batch['sensor']}")
    else:
        print("model_ready_patch_index.parquet not found. Please run 09_generate_model_ready_index.py first.")
