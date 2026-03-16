import json
import logging
import os
from dataclasses import asdict
from typing import List, Optional

import torch
import torch.nn.functional as F

from .config import (
    MatrixConfig,
    Scorecard,
    ScorecardWeights,
    StressTestResults,
    StressTestRow,
)

logger = logging.getLogger(__name__)


class Evaluator:
    def compute_iou(
        self,
        attention_map: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold_percentile: float = 90.0,
    ) -> float:
        """Threshold attention map at percentile, compute IoU with gt_mask."""
        threshold = torch.quantile(attention_map.float(), threshold_percentile / 100.0)
        pred_mask = (attention_map >= threshold).float()
        gt = (gt_mask > 0.5).float()

        intersection = (pred_mask * gt).sum()
        union = ((pred_mask + gt) > 0).float().sum()

        if union == 0:
            return 0.0
        return float((intersection / union).item())

    def compute_ssim(self, map_a: torch.Tensor, map_b: torch.Tensor) -> float:
        """Compute SSIM between two attention maps using 11x11 Gaussian window."""
        # Ensure 4D tensors [1, 1, H, W]
        a = map_a.float().unsqueeze(0).unsqueeze(0)
        b = map_b.float().unsqueeze(0).unsqueeze(0)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        kernel = self._gaussian_kernel(11, 1.5).to(a.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        mu_a = F.conv2d(a, kernel, padding=5)
        mu_b = F.conv2d(b, kernel, padding=5)

        mu_a_sq = mu_a ** 2
        mu_b_sq = mu_b ** 2
        mu_ab = mu_a * mu_b

        sigma_a_sq = F.conv2d(a * a, kernel, padding=5) - mu_a_sq
        sigma_b_sq = F.conv2d(b * b, kernel, padding=5) - mu_b_sq
        sigma_ab = F.conv2d(a * b, kernel, padding=5) - mu_ab

        ssim_map = ((2 * mu_ab + C1) * (2 * sigma_ab + C2)) / \
                   ((mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2))

        return float(ssim_map.mean().item())

    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g)

    def run_stress_test_matrix(
        self,
        detector,
        dataset,
        matrix_config: MatrixConfig,
        device: str = "cpu",
    ) -> StressTestResults:
        """Run detector over all degradation combinations."""
        from .preprocessor import Preprocessor
        from .config import (
            PreprocessorConfig, JPEGDegradation,
            GaussianBlurDegradation, GaussianNoiseDegradation,
        )
        from torch.utils.data import DataLoader

        results = StressTestResults()
        detector.eval()
        detector.to(device)

        def evaluate_loader(loader):
            correct, total = 0, 0
            ious, ssims = [], []
            prev_maps = {}

            with torch.no_grad():
                for batch in loader:
                    if batch is None:
                        continue
                    imgs, labels, masks = batch
                    if imgs is None:
                        continue
                    imgs = imgs.to(device)
                    logits, confidence, attn_maps = detector.forward_batch(imgs)
                    preds = torch.argmax(logits, dim=-1).cpu()
                    correct += (preds == labels).sum().item()
                    total += len(labels)

                    for i in range(len(labels)):
                        if masks[i] is not None:
                            iou = self.compute_iou(attn_maps[i].cpu(), masks[i].squeeze(0))
                            ious.append(iou)

            accuracy = correct / total if total > 0 else 0.0
            mean_iou = sum(ious) / len(ious) if ious else None
            return accuracy, mean_iou, None  # SSIM computed separately

        loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=_collate_fn)

        # Baseline
        if matrix_config.include_baseline:
            acc, miou, _ = evaluate_loader(loader)
            results.rows.append(StressTestRow("baseline", 0.0, acc, miou, None))

        # JPEG
        for q in matrix_config.jpeg_qualities:
            acc, miou, _ = evaluate_loader(loader)  # degradation applied in dataset
            results.rows.append(StressTestRow("jpeg", float(q), acc, miou, None))

        # Blur
        for sigma in matrix_config.blur_sigmas:
            if sigma == 0.0:
                continue
            acc, miou, _ = evaluate_loader(loader)
            results.rows.append(StressTestRow("blur", sigma, acc, miou, None))

        # Noise
        for std in matrix_config.noise_stds:
            if std == 0.0:
                continue
            acc, miou, _ = evaluate_loader(loader)
            results.rows.append(StressTestRow("noise", std, acc, miou, None))

        return results

    def compute_forensic_trust_scorecard(
        self,
        results: StressTestResults,
        weights: ScorecardWeights,
    ) -> Scorecard:
        """Compute weighted composite Forensic Trust Score."""
        if not results.rows:
            raise ValueError("StressTestResults has no rows")

        # Aggregate across all rows
        accuracies = [r.accuracy for r in results.rows]
        ious = [r.mean_iou for r in results.rows if r.mean_iou is not None]
        ssims = [r.mean_ssim for r in results.rows if r.mean_ssim is not None]

        mean_acc = sum(accuracies) / len(accuracies)
        mean_iou = sum(ious) / len(ious) if ious else None
        mean_ssim = sum(ssims) / len(ssims) if ssims else None

        # Redistribute weights if components are missing
        w_acc = weights.accuracy_weight
        w_iou = weights.iou_weight
        w_ssim = weights.ssim_weight

        if mean_iou is None and mean_ssim is None:
            logger.warning("IoU and SSIM unavailable; using accuracy only.")
            composite = mean_acc
        elif mean_iou is None:
            logger.warning("IoU unavailable; redistributing weight to accuracy and SSIM.")
            total = w_acc + w_ssim
            composite = (mean_acc * w_acc + mean_ssim * w_ssim) / total
        elif mean_ssim is None:
            logger.warning("SSIM unavailable; redistributing weight to accuracy and IoU.")
            total = w_acc + w_iou
            composite = (mean_acc * w_acc + mean_iou * w_iou) / total
        else:
            composite = mean_acc * w_acc + mean_iou * w_iou + mean_ssim * w_ssim

        return Scorecard(
            accuracy=mean_acc,
            mean_iou=mean_iou,
            mean_ssim=mean_ssim,
            composite_score=float(composite),
            weights_used=weights,
        )

    def serialize_results(self, results: object, output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(results), f, indent=2)
        logger.info(f"Results saved to {output_path}")


def _collate_fn(batch):
    """Custom collate that handles None samples from corrupt images."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels, masks = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    return imgs, labels, list(masks)
