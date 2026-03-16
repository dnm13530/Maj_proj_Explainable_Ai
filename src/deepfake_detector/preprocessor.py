import io
import logging
import os
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import (
    DegradationSpec,
    GaussianBlurDegradation,
    GaussianNoiseDegradation,
    JPEGDegradation,
    PreprocessorConfig,
)

logger = logging.getLogger(__name__)


class FaceDataset(Dataset):
    """PyTorch Dataset yielding (image_tensor, label, mask_or_None) tuples."""

    def __init__(self, samples: list, transform: Callable, mask_transform: Callable):
        self.samples = samples  # list of (img_path, label, mask_path_or_None)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Skipping corrupt image {img_path}: {e}")
            return None

        image_tensor = self.transform(image)

        mask_tensor = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert("L")
                mask_tensor = self.mask_transform(mask)
            except Exception as e:
                logger.warning(f"Could not load mask {mask_path}: {e}")

        return image_tensor, label, mask_tensor


class Preprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(config.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(config.target_size),
            transforms.ToTensor(),
        ])

    def load_dataset(self, root_dir: str, split: str) -> Dataset:
        """Load FaceForensics++ or Celeb-DF from directory structure."""
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        samples = []
        if self.config.dataset_type == "ff++":
            samples = self._load_ffpp(root_dir, split)
        elif self.config.dataset_type == "celeb-df":
            samples = self._load_celebdf(root_dir, split)
        else:
            raise ValueError(f"Unknown dataset_type: {self.config.dataset_type}")

        return FaceDataset(samples, self.transform, self.mask_transform)

    def _load_ffpp(self, root_dir: str, split: str) -> list:
        """
        Expected FF++ structure:
          root_dir/
            real/   (or original_sequences/)
            fake/   (or manipulated_sequences/)
            masks/  (optional)
        """
        samples = []
        for label_val, folder in [(0, "real"), (1, "fake")]:
            folder_path = os.path.join(root_dir, split, folder)
            if not os.path.exists(folder_path):
                # try alternate naming
                folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                logger.warning(f"Folder not found, skipping: {folder_path}")
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder_path, fname)
                    mask_path = os.path.join(root_dir, "masks", fname) if label_val == 1 else None
                    samples.append((img_path, label_val, mask_path))
        return samples

    def _load_celebdf(self, root_dir: str, split: str) -> list:
        """
        Expected Celeb-DF structure:
          root_dir/
            Celeb-real/
            Celeb-synthesis/
        """
        samples = []
        for label_val, folder in [(0, "Celeb-real"), (1, "Celeb-synthesis")]:
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                logger.warning(f"Folder not found, skipping: {folder_path}")
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder_path, fname)
                    samples.append((img_path, label_val, None))
        return samples

    def apply_degradation(self, image: Image.Image, degradation: DegradationSpec) -> Image.Image:
        """Apply a single degradation to a PIL image. Pure function."""
        if isinstance(degradation, JPEGDegradation):
            if not (1 <= degradation.quality <= 95):
                raise ValueError(f"JPEG quality must be in [1, 95], got {degradation.quality}")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=degradation.quality)
            buffer.seek(0)
            return Image.open(buffer).copy()

        elif isinstance(degradation, GaussianBlurDegradation):
            if degradation.kernel_size < 1 or degradation.kernel_size % 2 == 0:
                raise ValueError(f"Blur kernel_size must be odd and >= 1, got {degradation.kernel_size}")
            if degradation.sigma <= 0:
                raise ValueError(f"Blur sigma must be > 0, got {degradation.sigma}")
            from torchvision.transforms.functional import gaussian_blur
            tensor = transforms.ToTensor()(image)
            blurred = gaussian_blur(tensor, degradation.kernel_size, degradation.sigma)
            return transforms.ToPILImage()(blurred)

        elif isinstance(degradation, GaussianNoiseDegradation):
            if degradation.std < 0:
                raise ValueError(f"Noise std must be >= 0, got {degradation.std}")
            tensor = transforms.ToTensor()(image)
            noise = torch.randn_like(tensor) * degradation.std
            noisy = torch.clamp(tensor + noise, 0.0, 1.0)
            return transforms.ToPILImage()(noisy)

        else:
            raise ValueError(f"Unknown degradation type: {type(degradation)}")

    def build_degradation_pipeline(self, specs: List[DegradationSpec]) -> Callable:
        """Compose degradations in fixed order: JPEG → blur → noise."""
        # Sort by type priority
        ordered = sorted(specs, key=lambda s: (
            0 if isinstance(s, JPEGDegradation) else
            1 if isinstance(s, GaussianBlurDegradation) else 2
        ))

        def pipeline(image: Image.Image) -> Image.Image:
            for spec in ordered:
                image = self.apply_degradation(image, spec)
            return image

        return pipeline
