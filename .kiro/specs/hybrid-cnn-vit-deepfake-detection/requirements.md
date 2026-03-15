# Requirements Document

## Introduction

This document specifies the requirements for a Self-Explaining Hybrid CNN-ViT Framework for Deepfake Detection. The system combines a lightweight Convolutional Neural Network (CNN) backbone with a Vision Transformer (ViT) to detect AI-generated facial imagery (deepfakes). The primary innovation is the use of native ViT self-attention weights to generate interpretable "Artifact-Attention Maps" without relying on post-hoc XAI tools. The framework also introduces a rigorous evaluation protocol — the "Stress-Test Matrix" and "Forensic Trust Scorecard" — to assess both detection accuracy and explanation faithfulness under real-world image degradations such as JPEG compression, blur, and noise.

The system targets Technology Readiness Level (TRL) 3–4 and is benchmarked against FaceForensics++ and Celeb-DF datasets. It is developed in Python using PyTorch.

---

## Glossary

- **CNN**: Convolutional Neural Network — extracts local, microscopic forensic artifacts from image patches.
- **ViT**: Vision Transformer — processes global spatial inconsistencies via self-attention over image patches.
- **CLS Token**: The classification token in a ViT whose attention weights aggregate global context for the final prediction.
- **Artifact-Attention Map**: A spatial heatmap derived from the ViT's native [CLS] self-attention weights, highlighting regions the model considers forensically suspicious.
- **Deepfake**: An AI-generated or AI-manipulated facial image or video frame.
- **FaceForensics++**: A benchmark dataset of manipulated face videos with ground-truth manipulation masks.
- **Celeb-DF**: A high-quality deepfake video dataset used for evaluation.
- **Stress-Test Matrix**: A structured evaluation protocol applying controlled degradations (JPEG compression, Gaussian blur, additive noise) to test images before inference.
- **Forensic Trust Scorecard**: A composite metric combining detection accuracy, explanation faithfulness (IoU), and explanation stability (SSIM) across degradation levels.
- **IoU**: Intersection over Union — measures spatial overlap between the Artifact-Attention Map and the ground-truth manipulation mask.
- **SSIM**: Structural Similarity Index Measure — measures perceptual similarity between two images or maps.
- **Faithfulness**: The degree to which an Artifact-Attention Map correctly localizes the manipulated region as defined by the ground-truth mask.
- **Stability**: The degree to which Artifact-Attention Maps remain consistent across different degradation levels of the same image.
- **Degradation**: Any post-processing applied to an image that reduces quality, including JPEG compression, Gaussian blur, and additive Gaussian noise.
- **Detector**: The full hybrid CNN-ViT model responsible for classifying an image as real or deepfake.
- **Evaluator**: The component responsible for computing the Forensic Trust Scorecard metrics.
- **Preprocessor**: The component responsible for loading, resizing, normalizing, and applying degradations to input images.
- **Pretty_Printer**: The component responsible for rendering Artifact-Attention Maps as visual overlays on input images.

---

## Requirements

### Requirement 1: Data Ingestion and Preprocessing

**User Story:** As a researcher, I want to load and preprocess face images from standard benchmark datasets, so that I can feed consistent, normalized inputs into the detection model.

#### Acceptance Criteria

1. WHEN a dataset directory path is provided, THE Preprocessor SHALL load all image files and their associated labels (real or deepfake) and ground-truth manipulation masks where available.
2. THE Preprocessor SHALL resize all images to a configurable target resolution (default 224×224 pixels).
3. THE Preprocessor SHALL normalize pixel values using dataset-specific mean and standard deviation parameters.
4. WHEN a ground-truth manipulation mask is unavailable for an image, THE Preprocessor SHALL assign a null mask and exclude that image from faithfulness metric computation.
5. THE Preprocessor SHALL support loading from FaceForensics++ and Celeb-DF dataset directory structures.

---

### Requirement 2: Degradation Pipeline

**User Story:** As a researcher, I want to apply controlled image degradations to test images, so that I can evaluate model robustness under real-world distribution shifts.

#### Acceptance Criteria

1. THE Preprocessor SHALL support JPEG compression degradation at configurable quality levels (integer values from 1 to 95 inclusive).
2. THE Preprocessor SHALL support Gaussian blur degradation with a configurable kernel size (odd integer ≥ 1) and sigma value (float > 0).
3. THE Preprocessor SHALL support additive Gaussian noise degradation with a configurable standard deviation (float ≥ 0).
4. WHEN a degradation level of zero intensity is applied (e.g., JPEG quality 95, blur sigma 0, noise std 0), THE Preprocessor SHALL produce an output image that is perceptually equivalent to the original input.
5. WHEN multiple degradation types are specified, THE Preprocessor SHALL apply them sequentially in the order: JPEG compression → Gaussian blur → additive noise.
6. THE Preprocessor SHALL apply degradations only to test-split images and SHALL NOT modify training-split images.

---

### Requirement 3: Hybrid CNN-ViT Model Architecture

**User Story:** As a researcher, I want a hybrid model that combines local CNN feature extraction with global ViT attention, so that the model captures both microscopic artifacts and global spatial inconsistencies.

#### Acceptance Criteria

1. THE Detector SHALL include a CNN backbone that accepts a batch of normalized image tensors and produces a spatial feature map of configurable channel depth.
2. THE Detector SHALL include a ViT module that accepts the CNN feature map as a sequence of patch embeddings and produces per-head self-attention weight matrices for each transformer layer.
3. THE Detector SHALL include a classification head that accepts the [CLS] token embedding from the final ViT layer and produces a binary output (real or deepfake) with an associated confidence score in the range [0.0, 1.0].
4. WHEN a forward pass is executed, THE Detector SHALL return both the classification output and the raw self-attention weight tensors from the final ViT layer.
5. THE Detector SHALL be configurable with respect to CNN backbone depth, number of ViT layers, number of attention heads, and patch size.
6. THE Detector SHALL be implemented in PyTorch and SHALL be serializable to and deserializable from a checkpoint file without loss of model weights.

---

### Requirement 4: Artifact-Attention Map Generation

**User Story:** As a forensic analyst, I want the model to generate a spatial heatmap from its own attention weights, so that I can visually verify which image regions the model considers suspicious without relying on external XAI tools.

#### Acceptance Criteria

1. WHEN a forward pass produces self-attention weights from the final ViT layer, THE Detector SHALL compute the Artifact-Attention Map by averaging the [CLS] token's attention weights across all attention heads.
2. THE Detector SHALL reshape the averaged attention weights from the sequence dimension back to the spatial grid dimensions corresponding to the input patch layout.
3. THE Detector SHALL upsample the spatial attention grid to match the original input image resolution using bilinear interpolation.
4. THE Detector SHALL normalize the upsampled attention map to the range [0.0, 1.0] by dividing by the maximum value, provided the maximum value is greater than zero.
5. IF the maximum attention value is zero, THEN THE Detector SHALL return a zero-valued attention map of the correct spatial dimensions.
6. THE Pretty_Printer SHALL render the Artifact-Attention Map as a color-coded overlay on the original input image and SHALL save the result to a specified output path.

---

### Requirement 5: Faithfulness Evaluation (IoU)

**User Story:** As a researcher, I want to measure how accurately the Artifact-Attention Map localizes the manipulated region, so that I can quantify the forensic reliability of the model's explanations.

#### Acceptance Criteria

1. WHEN an Artifact-Attention Map and a binary ground-truth manipulation mask are provided, THE Evaluator SHALL compute the IoU score by thresholding the attention map at a configurable percentile (default 90th percentile) to produce a binary prediction mask.
2. THE Evaluator SHALL compute IoU as the ratio of the pixel-wise intersection to the pixel-wise union of the binary prediction mask and the ground-truth mask.
3. IF the union of the binary prediction mask and the ground-truth mask contains zero pixels, THEN THE Evaluator SHALL return an IoU score of 0.0.
4. THE Evaluator SHALL aggregate per-image IoU scores into a mean IoU value across all evaluated images.
5. WHEN computing mean IoU, THE Evaluator SHALL exclude images for which no ground-truth mask is available.

---

### Requirement 6: Stability Evaluation (SSIM)

**User Story:** As a researcher, I want to measure how consistently the Artifact-Attention Map behaves across different degradation levels of the same image, so that I can assess whether the model's explanations are stable under distribution shift.

#### Acceptance Criteria

1. WHEN Artifact-Attention Maps are generated for the same image at two different degradation levels, THE Evaluator SHALL compute the SSIM score between the two maps.
2. THE Evaluator SHALL use a Gaussian window of size 11×11 and sigma 1.5 for SSIM computation, consistent with the standard Wang et al. (2004) formulation.
3. THE Evaluator SHALL aggregate per-image SSIM scores into a mean SSIM value across all evaluated image pairs.
4. WHEN computing mean SSIM, THE Evaluator SHALL only include image pairs where both degradation levels produced valid (non-null) attention maps.

---

### Requirement 7: Stress-Test Matrix Execution

**User Story:** As a researcher, I want to run a structured evaluation across a predefined matrix of degradation types and severity levels, so that I can systematically characterize model performance under distribution shift.

#### Acceptance Criteria

1. THE Evaluator SHALL execute the Stress-Test Matrix by evaluating the Detector on each combination of degradation type (JPEG, blur, noise) and severity level defined in a configurable evaluation configuration file.
2. WHEN the Stress-Test Matrix is executed, THE Evaluator SHALL record detection accuracy, mean IoU, and mean SSIM for each degradation type and severity level combination.
3. THE Evaluator SHALL output the Stress-Test Matrix results as a structured data file (JSON format) to a configurable output directory.
4. WHEN a degradation type is set to baseline (no degradation), THE Evaluator SHALL include a baseline row in the Stress-Test Matrix results.

---

### Requirement 8: Forensic Trust Scorecard

**User Story:** As a forensic analyst, I want a single composite score that summarizes the model's detection accuracy, explanation faithfulness, and explanation stability, so that I can make an informed decision about deploying the model in a forensic context.

#### Acceptance Criteria

1. THE Evaluator SHALL compute the Forensic Trust Score as a weighted sum of normalized detection accuracy, mean IoU, and mean SSIM, using configurable weights that sum to 1.0.
2. WHEN the Forensic Trust Score is computed, THE Evaluator SHALL output the individual component scores alongside the composite score.
3. THE Evaluator SHALL serialize the Forensic Trust Scorecard to a JSON file at a configurable output path.
4. IF any component metric is unavailable (e.g., no ground-truth masks for IoU), THEN THE Evaluator SHALL redistribute the weight of the missing component proportionally among the remaining components and SHALL log a warning.

---

### Requirement 9: Model Training

**User Story:** As a researcher, I want to train the hybrid CNN-ViT model on labeled deepfake datasets, so that the model learns to distinguish real from manipulated facial images.

#### Acceptance Criteria

1. THE Detector SHALL be trainable using binary cross-entropy loss on labeled real/deepfake image pairs.
2. WHEN training is initiated, THE Detector SHALL support configurable hyperparameters including learning rate, batch size, number of epochs, and optimizer type (SGD or Adam).
3. THE Detector SHALL log training loss and validation accuracy at the end of each epoch to a configurable log file.
4. WHEN validation accuracy does not improve for a configurable number of consecutive epochs (patience), THE Detector SHALL halt training early and restore the best-performing checkpoint.
5. THE Detector SHALL save a model checkpoint after each epoch in which validation accuracy improves.

---

### Requirement 10: Inference and Reporting

**User Story:** As a forensic analyst, I want to run the trained model on new images and receive a structured report, so that I can audit the model's decisions with a verifiable evidence trail.

#### Acceptance Criteria

1. WHEN a trained model checkpoint and an input image are provided, THE Detector SHALL produce a classification label (real or deepfake), a confidence score, and an Artifact-Attention Map.
2. THE Evaluator SHALL produce a per-image inference report containing the classification label, confidence score, attention map file path, and IoU score (if a ground-truth mask is provided).
3. THE Evaluator SHALL serialize the per-image inference report to a JSON file at a configurable output path.
4. WHEN batch inference is performed on a directory of images, THE Evaluator SHALL produce one inference report per image and one aggregate summary report.
