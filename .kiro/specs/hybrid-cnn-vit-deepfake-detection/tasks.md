# Implementation Plan: Self-Explaining Hybrid CNN-ViT Deepfake Detection

## Overview

Incremental implementation of the hybrid CNN-ViT deepfake detection framework in Python/PyTorch. Each task builds on the previous, ending with a fully wired system. Tasks are ordered to validate core functionality early.

## Tasks

- [ ] 1. Project setup and core data models
  - Create the package structure: `src/deepfake_detector/` with `__init__.py`, `config.py`, `models.py`, `preprocessor.py`, `evaluator.py`, `pretty_printer.py`, `train.py`, `infer.py`
  - Implement all dataclasses in `config.py`: `PreprocessorConfig`, `ModelConfig`, `MatrixConfig`, `ScorecardWeights`, `StressTestRow`, `StressTestResults`, `Scorecard`, `InferenceReport`, `DetectorOutput`
  - Implement `DegradationSpec` tagged union types: `JPEGDegradation`, `GaussianBlurDegradation`, `GaussianNoiseDegradation`
  - Add `ScorecardWeights` post-init validation: raise `ValueError` if weights do not sum to 1.0 within 1e-6
  - Set up `pytest` and `hypothesis` as test dependencies (`requirements-dev.txt`)
  - _Requirements: 2.1, 2.2, 2.3, 3.5, 8.1_

- [ ] 2. Preprocessor — data loading and normalization
  - [ ] 2.1 Implement `Preprocessor.load_dataset` for FaceForensics++ and Celeb-DF directory structures
    - Return a PyTorch `Dataset` yielding `(image_tensor, label, mask_or_None)` tuples
    - Assign `None` mask when ground-truth mask file is absent
    - Raise `FileNotFoundError` for missing root directories
    - Log and skip corrupt image files
    - _Requirements: 1.1, 1.4, 1.5_
  - [ ] 2.2 Implement `Preprocessor` resize and normalization transforms
    - Resize to `PreprocessorConfig.target_size` using `torchvision.transforms`
    - Normalize using configurable mean/std
    - _Requirements: 1.2, 1.3_
  - [ ]* 2.3 Write property test for resize invariant
    - **Property 3 (partial): For any PIL image of any size, after preprocessing the output tensor spatial dimensions equal the configured target size**
    - **Validates: Requirements 1.2**

- [ ] 3. Preprocessor — degradation pipeline
  - [ ] 3.1 Implement `Preprocessor.apply_degradation` for JPEG, Gaussian blur, and Gaussian noise
    - JPEG: use `PIL` save/load with quality parameter; validate quality in [1, 95], raise `ValueError` otherwise
    - Blur: use `torchvision.transforms.functional.gaussian_blur`; validate odd kernel size ≥ 1 and sigma > 0
    - Noise: add `torch.randn`-scaled tensor; validate std ≥ 0
    - _Requirements: 2.1, 2.2, 2.3_
  - [ ] 3.2 Implement `Preprocessor.build_degradation_pipeline`
    - Compose degradations in fixed order: JPEG → blur → noise
    - Return a callable transform
    - Enforce that degradations are only applied to test-split data
    - _Requirements: 2.5, 2.6_
  - [ ]* 3.3 Write property test for degradation identity (Property 1)
    - **Property 1: For any valid input image, applying zero-intensity degradation produces output within tolerance ≤ 5/255 per channel**
    - **Validates: Requirements 2.4**
  - [ ]* 3.4 Write property test for degradation pipeline ordering (Property 2)
    - **Property 2: For any image and degradation specs, pipeline result equals sequential individual application in JPEG → blur → noise order**
    - **Validates: Requirements 2.5**

- [ ] 4. Checkpoint — preprocessor complete
  - Ensure all preprocessor tests pass, ask the user if questions arise.

- [ ] 5. CNN backbone
  - [ ] 5.1 Implement `CNNBackbone` in `models.py`
    - 4-layer ConvNet: Conv2d → BatchNorm2d → ReLU blocks, configurable output channel depth (`ModelConfig.cnn_out_channels`)
    - Accept input `[B, 3, H, W]`, produce feature map `[B, C, H', W']`
    - _Requirements: 3.1_
  - [ ]* 5.2 Write property test for CNN output shape (Property 3, partial)
    - **Property 3 (partial): For any valid batch tensor [B,3,H,W] and model config, CNN backbone output has shape [B, cnn_out_channels, H', W']**
    - **Validates: Requirements 3.1**

- [ ] 6. ViT module with attention capture
  - [ ] 6.1 Implement `ViTModule` in `models.py`
    - Flatten CNN feature map to patch token sequence `[B, N, D]`
    - Prepend learnable [CLS] token
    - Stack `ModelConfig.vit_num_layers` Transformer encoder layers with `ModelConfig.vit_num_heads` heads
    - Register a forward hook on the final `nn.MultiheadAttention` to capture attention weights `[B, num_heads, seq_len, seq_len]`
    - _Requirements: 3.2, 3.4_
  - [ ]* 6.2 Write property test for ViT attention weight shape (Property 3, partial)
    - **Property 3 (partial): For any valid feature map, ViT produces attention weights of shape [B, num_heads, seq_len, seq_len]**
    - **Validates: Requirements 3.2**

- [ ] 7. Classification head and full forward pass
  - [ ] 7.1 Implement `ClassificationHead` in `models.py`
    - Accept [CLS] token embedding from final ViT layer
    - Produce logits `[B, 2]` and confidence score `[B]` via softmax
    - _Requirements: 3.3_
  - [ ] 7.2 Implement `HybridCNNViT.forward` wiring CNN → ViT → head
    - Return `DetectorOutput` with label, confidence, and raw attention weights
    - _Requirements: 3.4, 3.5_
  - [ ]* 7.3 Write property test for confidence score range (Property 5)
    - **Property 5: For any input image tensor, confidence score lies in [0.0, 1.0]**
    - **Validates: Requirements 3.3**
  - [ ] 7.4 Implement `HybridCNNViT.save_checkpoint` and `load_checkpoint`
    - Serialize/deserialize full model state dict to/from file
    - Raise `FileNotFoundError` if checkpoint path does not exist on load
    - _Requirements: 3.6_
  - [ ]* 7.5 Write property test for checkpoint round-trip (Property 11)
    - **Property 11: For any trained HybridCNNViT, save then load produces identical weight tensors**
    - **Validates: Requirements 3.6**

- [ ] 8. Artifact-Attention Map generation
  - [ ] 8.1 Implement `HybridCNNViT.generate_artifact_attention_map`
    - Extract [CLS] row from attention weights: index `[:, :, 0, 1:]` (CLS attends to all patch tokens)
    - Average across heads: `mean(dim=1)`
    - Reshape to spatial grid `[B, sqrt(N), sqrt(N)]`
    - Upsample to input resolution via `F.interpolate(..., mode='bilinear')`
    - Normalize to [0,1]: divide by max; return zero map if max == 0
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  - [ ]* 8.2 Write property test for attention map shape invariant (Property 3, full)
    - **Property 3: For any input tensor [B,3,H,W], generate_artifact_attention_map returns shape [B,H,W]**
    - **Validates: Requirements 4.3**
  - [ ]* 8.3 Write property test for attention map value range (Property 4)
    - **Property 4: For any input tensor, all attention map values lie in [0.0, 1.0]; edge case: all-zero attention weights return all-zero map**
    - **Validates: Requirements 4.4, 4.5**

- [ ] 9. Checkpoint — model complete
  - Ensure all model tests pass, ask the user if questions arise.

- [ ] 10. Evaluator — IoU computation
  - [ ] 10.1 Implement `Evaluator.compute_iou`
    - Threshold attention map at configurable percentile using `torch.quantile`
    - Compute pixel-wise intersection and union
    - Return IoU = intersection / union; return 0.0 if union == 0
    - _Requirements: 5.1, 5.2, 5.3_
  - [ ]* 10.2 Write property test for IoU correctness and range (Property 6)
    - **Property 6: For any attention map and ground-truth mask, IoU lies in [0.0, 1.0]; edge case: empty union returns 0.0**
    - **Validates: Requirements 5.2, 5.3**
  - [ ] 10.3 Implement `Evaluator` mean IoU aggregation with null mask exclusion
    - Compute mean over non-null entries only
    - _Requirements: 5.4, 5.5_
  - [ ]* 10.4 Write property test for null mask exclusion (Property 7)
    - **Property 7: For any dataset with mixed null/non-null masks, mean IoU excludes null-mask images**
    - **Validates: Requirements 5.5**

- [ ] 11. Evaluator — SSIM computation
  - [ ] 11.1 Implement `Evaluator.compute_ssim`
    - Use `torchmetrics.functional.structural_similarity_index_measure` or manual implementation with 11×11 Gaussian window, sigma=1.5
    - Aggregate mean SSIM excluding null pairs
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - [ ]* 11.2 Write property test for SSIM range and identity (Property 8)
    - **Property 8: For any pair of valid attention maps, SSIM lies in [-1.0, 1.0]; identical maps return SSIM = 1.0**
    - **Validates: Requirements 6.1**

- [ ] 12. Evaluator — Stress-Test Matrix and Forensic Trust Scorecard
  - [ ] 12.1 Implement `Evaluator.run_stress_test_matrix`
    - Iterate over all (degradation_type, severity) combinations from `MatrixConfig`
    - Include baseline row when `include_baseline=True`
    - Record accuracy, mean IoU, mean SSIM per row
    - Serialize results to JSON at configured output path
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [ ]* 12.2 Write property test for Stress-Test Matrix completeness (Property 12)
    - **Property 12: For any MatrixConfig with include_baseline=True, results contain exactly one baseline row**
    - **Validates: Requirements 7.4**
  - [ ] 12.3 Implement `Evaluator.compute_forensic_trust_scorecard`
    - Compute weighted sum: `accuracy * w_acc + mean_iou * w_iou + mean_ssim * w_ssim`
    - Redistribute weights proportionally when a component is unavailable; log warning
    - Serialize scorecard to JSON
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [ ]* 12.4 Write property test for scorecard weights invariant (Property 9)
    - **Property 9: For any ScorecardWeights, accuracy_weight + iou_weight + ssim_weight == 1.0 within 1e-6**
    - **Validates: Requirements 8.1**
  - [ ]* 12.5 Write property test for scorecard composite score range (Property 10)
    - **Property 10: For any valid StressTestResults and ScorecardWeights, composite score lies in [0.0, 1.0]**
    - **Validates: Requirements 8.1**

- [ ] 13. Checkpoint — evaluator complete
  - Ensure all evaluator tests pass, ask the user if questions arise.

- [ ] 14. Pretty_Printer
  - [ ] 14.1 Implement `Pretty_Printer.render_overlay`
    - Apply jet colormap to attention map using `matplotlib.cm`
    - Blend overlay with original image at configurable alpha
    - Save result to specified output path using PIL
    - _Requirements: 4.6_
  - [ ]* 14.2 Write example test for overlay file creation
    - Test that after calling `render_overlay`, the output file exists at the specified path
    - _Requirements: 4.6_

- [ ] 15. Training pipeline
  - [ ] 15.1 Implement `train.py` training loop
    - Binary cross-entropy loss on real/deepfake labels
    - Configurable optimizer (SGD or Adam), learning rate, batch size, epochs
    - Log training loss and validation accuracy per epoch to a log file
    - Save checkpoint when validation accuracy improves
    - Implement early stopping with configurable patience
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - [ ]* 15.2 Write property test for training log completeness
    - **For any training run of N epochs, the log file contains exactly N entries**
    - **Validates: Requirements 9.3**

- [ ] 16. Inference pipeline and reporting
  - [ ] 16.1 Implement `infer.py` single-image and batch inference
    - Load checkpoint, run forward pass, generate attention map, produce `DetectorOutput`
    - Generate per-image `InferenceReport` (label, confidence, attention map path, IoU if mask available)
    - Generate aggregate summary report for batch inference
    - Serialize reports to JSON
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  - [ ]* 16.2 Write property test for inference report serialization round-trip (Property 13)
    - **Property 13: For any InferenceReport, serializing to JSON then deserializing produces an object with identical field values**
    - **Validates: Requirements 10.3_
  - [ ]* 16.3 Write property test for batch report count invariant
    - **For any batch of N images, batch inference produces exactly N individual reports**
    - **Validates: Requirements 10.4**

- [ ] 17. Final checkpoint — full system integration
  - Wire all components together: Preprocessor → Detector → Evaluator → Pretty_Printer
  - Run a smoke test: load a small sample from FaceForensics++ (or mock data), train for 1 epoch, run inference, generate scorecard
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `@settings(max_examples=100)` minimum
- Each property test must include the comment: `# Feature: hybrid-cnn-vit-deepfake-detection, Property N: <property_text>`
- Checkpoints at tasks 4, 9, 13, 17 ensure incremental validation
