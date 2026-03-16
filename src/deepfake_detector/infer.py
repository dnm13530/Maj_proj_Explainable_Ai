import json
import logging
import os
from dataclasses import asdict
from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms

from .config import InferenceReport, PreprocessorConfig
from .evaluator import Evaluator
from .models import HybridCNNViT
from .pretty_printer import PrettyPrinter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_inference(
    checkpoint_path: str,
    image_path: str,
    output_dir: str = "outputs/inference",
    gt_mask_path: Optional[str] = None,
    pre_cfg: Optional[PreprocessorConfig] = None,
    device: str = "cpu",
) -> InferenceReport:
    """Run inference on a single image and return a structured report."""
    os.makedirs(output_dir, exist_ok=True)

    if pre_cfg is None:
        pre_cfg = PreprocessorConfig()

    model = HybridCNNViT.load_checkpoint(checkpoint_path).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(pre_cfg.target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_cfg.mean, std=pre_cfg.std),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        output = model(tensor)

    label_str = "deepfake" if output.label == 1 else "real"
    attn_map = output.attention_map  # [H, W]

    # Save attention overlay
    fname = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(output_dir, f"{fname}_overlay.png")
    printer = PrettyPrinter()
    printer.render_overlay(image, attn_map, overlay_path)

    # Compute IoU if mask provided
    iou_score = None
    if gt_mask_path and os.path.exists(gt_mask_path):
        mask = Image.open(gt_mask_path).convert("L")
        mask_transform = transforms.Compose([
            transforms.Resize(pre_cfg.target_size),
            transforms.ToTensor(),
        ])
        mask_tensor = mask_transform(mask).squeeze(0)
        evaluator = Evaluator()
        iou_score = evaluator.compute_iou(attn_map.cpu(), mask_tensor)

    report = InferenceReport(
        image_path=image_path,
        label=label_str,
        confidence=output.confidence,
        attention_map_path=overlay_path,
        iou_score=iou_score,
    )

    # Serialize report
    report_path = os.path.join(output_dir, f"{fname}_report.json")
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(f"Report saved to {report_path}")

    return report


def run_batch_inference(
    checkpoint_path: str,
    image_dir: str,
    output_dir: str = "outputs/inference",
    pre_cfg: Optional[PreprocessorConfig] = None,
    device: str = "cpu",
) -> List[InferenceReport]:
    """Run inference on all images in a directory."""
    supported = (".jpg", ".jpeg", ".png")
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(supported)
    ]

    reports = []
    for img_path in image_paths:
        try:
            report = run_inference(checkpoint_path, img_path, output_dir, pre_cfg=pre_cfg, device=device)
            reports.append(report)
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")

    # Aggregate summary
    if reports:
        n_fake = sum(1 for r in reports if r.label == "deepfake")
        n_real = len(reports) - n_fake
        avg_conf = sum(r.confidence for r in reports) / len(reports)
        valid_ious = [r.iou_score for r in reports if r.iou_score is not None]
        summary = {
            "total_images": len(reports),
            "deepfake_count": n_fake,
            "real_count": n_real,
            "avg_confidence": avg_conf,
            "mean_iou": sum(valid_ious) / len(valid_ious) if valid_ious else None,
        }
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Batch summary saved to {summary_path}")

    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--image_dir", default=None, help="Directory for batch inference")
    parser.add_argument("--output_dir", default="outputs/inference")
    parser.add_argument("--mask", default=None, help="Ground-truth mask path (single image only)")
    parser.add_argument("--dataset_type", default="ff++")
    args = parser.parse_args()

    pre_cfg = PreprocessorConfig(dataset_type=args.dataset_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.image:
        report = run_inference(args.checkpoint, args.image, args.output_dir, args.mask, pre_cfg, device)
        print(f"Label: {report.label} | Confidence: {report.confidence:.4f} | IoU: {report.iou_score}")
    elif args.image_dir:
        reports = run_batch_inference(args.checkpoint, args.image_dir, args.output_dir, pre_cfg, device)
        print(f"Processed {len(reports)} images.")
    else:
        print("Provide --image or --image_dir")
