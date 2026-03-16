"""Smoke tests — verify the full pipeline runs on synthetic data."""
import io
import torch
import pytest
from PIL import Image

from src.deepfake_detector.config import (
    ModelConfig, PreprocessorConfig,
    JPEGDegradation, GaussianBlurDegradation, GaussianNoiseDegradation,
    ScorecardWeights, StressTestResults, StressTestRow,
)
from src.deepfake_detector.models import HybridCNNViT
from src.deepfake_detector.preprocessor import Preprocessor
from src.deepfake_detector.evaluator import Evaluator
from src.deepfake_detector.pretty_printer import PrettyPrinter


@pytest.fixture
def model():
    cfg = ModelConfig(cnn_out_channels=32, vit_num_layers=1, vit_num_heads=2, vit_embed_dim=64)
    return HybridCNNViT(cfg)


@pytest.fixture
def dummy_image():
    return Image.fromarray((torch.rand(224, 224, 3) * 255).byte().numpy())


def test_forward_pass(model, dummy_image):
    pre = Preprocessor(PreprocessorConfig())
    tensor = pre.transform(dummy_image).unsqueeze(0)
    output = model(tensor)
    assert output.label in (0, 1)
    assert 0.0 <= output.confidence <= 1.0
    assert output.attention_map.shape == (224, 224)


def test_attention_map_range(model, dummy_image):
    pre = Preprocessor(PreprocessorConfig())
    tensor = pre.transform(dummy_image).unsqueeze(0)
    output = model(tensor)
    assert output.attention_map.min() >= 0.0
    assert output.attention_map.max() <= 1.0


def test_jpeg_degradation(dummy_image):
    pre = Preprocessor(PreprocessorConfig())
    result = pre.apply_degradation(dummy_image, JPEGDegradation(quality=50))
    assert isinstance(result, Image.Image)


def test_blur_degradation(dummy_image):
    pre = Preprocessor(PreprocessorConfig())
    result = pre.apply_degradation(dummy_image, GaussianBlurDegradation(kernel_size=5, sigma=1.0))
    assert isinstance(result, Image.Image)


def test_noise_degradation(dummy_image):
    pre = Preprocessor(PreprocessorConfig())
    result = pre.apply_degradation(dummy_image, GaussianNoiseDegradation(std=0.05))
    assert isinstance(result, Image.Image)


def test_invalid_jpeg_quality(dummy_image):
    pre = Preprocessor(PreprocessorConfig())
    with pytest.raises(ValueError):
        pre.apply_degradation(dummy_image, JPEGDegradation(quality=0))


def test_iou_range():
    evaluator = Evaluator()
    attn = torch.rand(224, 224)
    mask = (torch.rand(224, 224) > 0.5).float()
    iou = evaluator.compute_iou(attn, mask)
    assert 0.0 <= iou <= 1.0


def test_iou_empty_union():
    evaluator = Evaluator()
    attn = torch.zeros(224, 224)
    mask = torch.zeros(224, 224)
    assert evaluator.compute_iou(attn, mask) == 0.0


def test_ssim_identical_maps():
    evaluator = Evaluator()
    m = torch.rand(64, 64)
    ssim = evaluator.compute_ssim(m, m)
    assert abs(ssim - 1.0) < 1e-3


def test_scorecard_weights_invalid():
    with pytest.raises(ValueError):
        ScorecardWeights(accuracy_weight=0.5, iou_weight=0.5, ssim_weight=0.5)


def test_forensic_trust_scorecard():
    evaluator = Evaluator()
    results = StressTestResults(rows=[
        StressTestRow("baseline", 0.0, 0.85, 0.6, 0.7),
        StressTestRow("jpeg", 50.0, 0.75, 0.5, 0.6),
    ])
    weights = ScorecardWeights(0.4, 0.4, 0.2)
    scorecard = evaluator.compute_forensic_trust_scorecard(results, weights)
    assert 0.0 <= scorecard.composite_score <= 1.0


def test_checkpoint_roundtrip(model, tmp_path):
    path = str(tmp_path / "model.pt")
    model.save_checkpoint(path)
    loaded = HybridCNNViT.load_checkpoint(path)
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.allclose(v1, v2), f"Mismatch at {k1}"


def test_overlay_saved(model, dummy_image, tmp_path):
    pre = Preprocessor(PreprocessorConfig())
    tensor = pre.transform(dummy_image).unsqueeze(0)
    output = model(tensor)
    out_path = str(tmp_path / "overlay.png")
    PrettyPrinter().render_overlay(dummy_image, output.attention_map, out_path)
    import os
    assert os.path.exists(out_path)
