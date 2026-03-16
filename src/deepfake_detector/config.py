from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional


@dataclass
class PreprocessorConfig:
    target_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    dataset_type: str = "ff++"  # "ff++" | "celeb-df"


@dataclass
class ModelConfig:
    cnn_out_channels: int = 64
    patch_size: int = 16
    vit_num_layers: int = 2
    vit_num_heads: int = 4
    vit_embed_dim: int = 128
    num_classes: int = 2


@dataclass
class MatrixConfig:
    jpeg_qualities: List[int] = field(default_factory=lambda: [95, 75, 50, 25, 10])
    blur_sigmas: List[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 4.0])
    noise_stds: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])
    include_baseline: bool = True


@dataclass
class ScorecardWeights:
    accuracy_weight: float = 0.4
    iou_weight: float = 0.4
    ssim_weight: float = 0.2

    def __post_init__(self):
        total = self.accuracy_weight + self.iou_weight + self.ssim_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scorecard weights must sum to 1.0, got {total}")


@dataclass
class StressTestRow:
    degradation_type: str   # "baseline" | "jpeg" | "blur" | "noise"
    severity: float
    accuracy: float
    mean_iou: Optional[float]
    mean_ssim: Optional[float]


@dataclass
class StressTestResults:
    rows: List[StressTestRow] = field(default_factory=list)


@dataclass
class Scorecard:
    accuracy: float
    mean_iou: Optional[float]
    mean_ssim: Optional[float]
    composite_score: float
    weights_used: ScorecardWeights


@dataclass
class DetectorOutput:
    label: int           # 0 = real, 1 = deepfake
    confidence: float    # in [0.0, 1.0]
    attention_map: object  # Tensor [H, W]


@dataclass
class InferenceReport:
    image_path: str
    label: str           # "real" | "deepfake"
    confidence: float
    attention_map_path: str
    iou_score: Optional[float]


# Degradation specs
@dataclass
class JPEGDegradation:
    quality: int  # 1–95

@dataclass
class GaussianBlurDegradation:
    kernel_size: int  # odd, >= 1
    sigma: float      # > 0

@dataclass
class GaussianNoiseDegradation:
    std: float  # >= 0

DegradationSpec = Union[JPEGDegradation, GaussianBlurDegradation, GaussianNoiseDegradation]
