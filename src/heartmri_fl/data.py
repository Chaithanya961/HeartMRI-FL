"""Data loading utilities for heart MRI federated learning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


class HeartMRIClientData:
    def __init__(self, train_dataset: TensorDataset, val_dataset: TensorDataset) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=batch_size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = image - float(np.min(image))
    denom = float(np.max(image) - np.min(image))
    if denom > 0:
        image = image / denom
    return image


def resize_image(image: np.ndarray, image_size: Tuple[int, int]) -> torch.Tensor:
    if image.ndim == 2:
        # Single channel image (grayscale)
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=image_size, mode="bilinear", align_corners=False)
        return resized.squeeze(0).squeeze(0)  # Remove both batch and channel dims
    elif image.ndim == 3:
        # Multi-channel image
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(tensor, size=image_size, mode="bilinear", align_corners=False)
        return resized.squeeze(0)
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")


def load_dicom_slice(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(str(path), force=True)
    image = dataset.pixel_array.astype(np.float32)
    if hasattr(dataset, "RescaleSlope") and hasattr(dataset, "RescaleIntercept"):
        image = image * float(dataset.RescaleSlope) + float(dataset.RescaleIntercept)
    return normalize_image(image)


def load_nifti_volume(path: Path) -> np.ndarray:
    image = nib.load(str(path)).get_fdata()
    if image.ndim == 4:
        image = image[..., 0]
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    return normalize_image(np.asarray(image, dtype=np.float32))


def load_jpg_image(path: Path) -> np.ndarray:
    image = Image.open(str(path)).convert('L')  # Convert to grayscale
    image = np.array(image, dtype=np.float32)
    return normalize_image(image)


def extract_slices(volume: np.ndarray, max_slices: int = 16) -> List[np.ndarray]:
    if volume.ndim == 2:
        return [volume]

    if volume.ndim == 3:
        depth = volume.shape[2]
        indices = np.linspace(0, depth - 1, min(depth, max_slices), dtype=int)
        return [volume[:, :, idx] for idx in indices]

    raise ValueError("Unsupported volume shape for MRI data")


def split_file_paths(file_paths: List[Path], groups: int) -> List[List[Path]]:
    buckets: List[List[Path]] = [[] for _ in range(max(1, groups))]
    for index, path in enumerate(file_paths):
        buckets[index % len(buckets)].append(path)
    return buckets


def find_client_directories(root: Path) -> Dict[str, List[Path]]:
    clients: Dict[str, List[Path]] = {}
    if not root.exists():
        return clients

    for child in sorted(root.iterdir()):
        if child.is_dir():
            files = [
                path
                for path in child.rglob("*")
                if path.suffix.lower() in {".dcm", ".nii", ".nii.gz", ".jpg", ".jpeg"}
            ]
            if files:
                clients[child.name] = sorted(files)

    if not clients:
        files = [
            path
            for path in root.rglob("*")
            if path.suffix.lower() in {".dcm", ".nii", ".nii.gz", ".jpg", ".jpeg"}
        ]
        if files:
            clients["0"] = sorted(files)

    return clients


def build_client_data(
    root: Path,
    image_size: Tuple[int, int] = (128, 128),
    num_hospitals: int = 3,
    subclients_per_hospital: int = 1,
    num_classes: int = 2,
    samples_per_client: int = 32,
) -> Dict[str, HeartMRIClientData]:
    clients = find_client_directories(root)
    if not clients:
        return build_synthetic_client_data(
            num_hospitals=num_hospitals,
            subclients_per_hospital=subclients_per_hospital,
            image_size=image_size,
            num_classes=num_classes,
            samples_per_client=samples_per_client,
        )

    client_data: Dict[str, HeartMRIClientData] = {}
    for hospital_index, (_, file_paths) in enumerate(clients.items()):
        groups = split_file_paths(file_paths, subclients_per_hospital)
        for sub_index, group in enumerate(groups):
            if not group:
                continue

            samples: List[torch.Tensor] = []
            for file_path in group:
                if file_path.suffix.lower() == ".dcm":
                    slice_image = load_dicom_slice(file_path)
                    samples.append(resize_image(slice_image, image_size))
                elif file_path.suffix.lower() in {".jpg", ".jpeg"}:
                    slice_image = load_jpg_image(file_path)
                    samples.append(resize_image(slice_image, image_size))
                else:
                    volume = load_nifti_volume(file_path)
                    for slice_image in extract_slices(volume, max_slices=8):
                        samples.append(resize_image(slice_image, image_size))
                if len(samples) >= samples_per_client:
                    break

            if not samples:
                continue

            images = torch.stack(samples[:samples_per_client]).unsqueeze(1)
            labels = torch.zeros(len(images), dtype=torch.long)

            split = int(0.8 * len(images))
            if split < 1:
                train_dataset = TensorDataset(images, labels)
                val_dataset = TensorDataset(images, labels)
            else:
                train_dataset = TensorDataset(images[:split], labels[:split])
                val_dataset = TensorDataset(images[split:], labels[split:])

            client_data[f"{hospital_index}_{sub_index}"] = HeartMRIClientData(train_dataset, val_dataset)

    if not client_data:
        return build_synthetic_client_data(
            num_hospitals=num_hospitals,
            subclients_per_hospital=subclients_per_hospital,
            image_size=image_size,
            num_classes=num_classes,
            samples_per_client=samples_per_client,
        )

    return client_data


def build_synthetic_client_data(
    num_hospitals: int = 3,
    subclients_per_hospital: int = 1,
    image_size: Tuple[int, int] = (128, 128),
    num_classes: int = 2,
    samples_per_client: int = 32,
    normal_ratio: float = 0.8,
) -> Dict[str, HeartMRIClientData]:
    client_data: Dict[str, HeartMRIClientData] = {}
    for hospital_index in range(num_hospitals):
        hospital_shift = float(hospital_index) * 0.5
        for sub_index in range(subclients_per_hospital):
            normal_samples = max(1, int(samples_per_client * normal_ratio))
            anomaly_samples = samples_per_client - normal_samples

            normal_images = torch.randn(
                normal_samples, 1, image_size[0], image_size[1], dtype=torch.float32
            ) + hospital_shift
            anomaly_images = torch.randn(
                anomaly_samples, 1, image_size[0], image_size[1], dtype=torch.float32
            ) + hospital_shift + 3.0

            images = torch.cat([normal_images, anomaly_images], dim=0)
            labels = torch.cat(
                [torch.zeros(normal_samples, dtype=torch.long), torch.ones(anomaly_samples, dtype=torch.long)]
            )

            perm = torch.randperm(len(images))
            images = images[perm]
            labels = labels[perm]

            normal_mask = labels == 0
            train_images = images[normal_mask]
            train_labels = labels[normal_mask]
            if len(train_images) == 0:
                train_images = images[:1]
                train_labels = labels[:1]

            val_images = images
            val_labels = labels

            train_dataset = TensorDataset(train_images, train_labels)
            val_dataset = TensorDataset(val_images, val_labels)
            client_data[f"{hospital_index}_{sub_index}"] = HeartMRIClientData(train_dataset, val_dataset)

    return client_data
