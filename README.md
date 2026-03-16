# Self-Explaining Hybrid CNN-ViT Deepfake Detector

A prototype that detects AI-generated facial imagery (deepfakes) and explains *why* using native ViT self-attention weights — no Grad-CAM, no LIME. Evaluated under real-world image degradations (JPEG compression, blur, noise) via a Stress-Test Matrix and Forensic Trust Scorecard.

> Aligns with SDG 16: Peace, Justice, and Strong Institutions.

---

## Repo Structure

```
src/deepfake_detector/
    config.py          — all dataclasses and config objects
    models.py          — HybridCNNViT, CNNBackbone, ViTModule, ClassificationHead
    preprocessor.py    — data loading, normalization, degradation pipeline
    evaluator.py       — IoU, SSIM, Stress-Test Matrix, Forensic Trust Scorecard
    pretty_printer.py  — attention map overlay visualization
    train.py           — training loop with early stopping and checkpointing
    infer.py           — single-image and batch inference with JSON reports

notebooks/
    deepfake_detection_colab.ipynb  — end-to-end Colab/Kaggle notebook

tests/
    test_smoke.py      — 13 smoke tests covering all components

.kiro/specs/hybrid-cnn-vit-deepfake-detection/
    requirements.md    — full requirements with acceptance criteria
    design.md          — architecture, interfaces, correctness properties
    tasks.md           — implementation task breakdown
```

---

## Environment Setup

```bash
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # + pytest + hypothesis for testing
```

Run tests to verify everything works before touching the dataset:

```bash
python -m pytest tests/test_smoke.py -v
```

All 13 tests should pass.

---

## To Do After Getting the Dataset

### Step 1 — Organize the dataset (Day 1)

Structure your FF++ frames exactly like this so the preprocessor loads them automatically:

```
datasets/ff++/
    train/
        real/        ← frames extracted from original videos
        fake/        ← frames extracted from manipulated videos
    test/
        real/
        fake/
        masks/       ← ground-truth manipulation masks (FF++ provides these as PNGs)
```

For Celeb-DF, use this structure instead:

```
datasets/celeb-df/
    Celeb-real/
    Celeb-synthesis/
```

**Tips:**
- Use the FF++ download script with `--num_videos 1000` and `-c c23` (compressed) to keep size manageable on Colab/Kaggle.
- Extract frames at 1 fps to avoid redundant near-identical frames — `ffmpeg -i video.mp4 -vf fps=1 frames/%04d.jpg`
- Store the organized dataset on Google Drive (Colab) or as a Kaggle Dataset so it persists across sessions.

---

### Step 2 — Train the model on Colab (Week 1–2)

1. Upload `notebooks/deepfake_detection_colab.ipynb` to Google Colab.
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Mount Google Drive and update `DATASET_ROOT` to point at your dataset folder.
4. Run all cells top to bottom.

Monitor `outputs/train_log.csv` after each epoch. You want:
- Validation accuracy climbing above **70%** by epoch 10.
- Loss decreasing steadily — if it plateaus early, try lowering `learning_rate` to `5e-5`.

The best checkpoint is auto-saved to `SAVE_DIR/best_model.pt` whenever validation accuracy improves. **This file is your most important output — back it up to Drive immediately.**

If Colab disconnects mid-training (it will), just re-run from the training cell — the checkpoint from the last best epoch is already saved.

---

### Step 3 — Run the Stress-Test Matrix (Week 2–3)

This is the core research contribution. The notebook has this wired up in the final cells.

It evaluates the trained model across every combination of:
- JPEG quality: 95, 75, 50, 25, 10
- Gaussian blur sigma: 1.0, 2.0, 4.0
- Gaussian noise std: 0.05, 0.1, 0.2
- Baseline: no degradation

For each combination it records **accuracy**, **mean IoU** (explanation faithfulness), and **mean SSIM** (explanation stability).

Output files saved to `SAVE_DIR/`:
- `stress_test_matrix.json` — full results table
- `forensic_trust_scorecard.json` — composite score breakdown
- `stress_test_plot.png` — accuracy + IoU degradation curve chart

---

### Step 4 — Analyze and interpret results (Week 3–4)

This is where your team's research contribution lives. Work through these questions using the JSON output and plots:

- At what JPEG quality does accuracy drop below 60%? Below 50%?
- Does IoU (explanation faithfulness) degrade faster or slower than accuracy under compression?
- Which degradation type is most damaging — JPEG, blur, or noise?
- Is there a degradation level where accuracy stays high but IoU collapses? (This is a key finding — the model is still "right" but for the wrong reasons.)
- Compare your Forensic Trust Score at baseline vs. worst degradation — what is the drop?

These answers become your **Results and Discussion** section in the paper.


## Running Inference on a New Image

```bash
python -m src.deepfake_detector.infer \
    --checkpoint outputs/best_model.pt \
    --image path/to/face.jpg \
    --output_dir outputs/inference
```

Output: a JSON report + attention map overlay saved to `outputs/inference/`.

For a full directory:

```bash
python -m src.deepfake_detector.infer \
    --checkpoint outputs/best_model.pt \
    --image_dir path/to/images/ \
    --output_dir outputs/inference
```

---

## Key Design Decisions

- **Native attention over post-hoc XAI** — Artifact-Attention Maps are extracted directly from the ViT's [CLS] token attention weights during the forward pass. No Grad-CAM, no LIME, no extra compute.
- **Lightweight model for free-tier GPUs** — default config fits comfortably in a Colab T4 (12GB VRAM) with batch size 16.
- **Degradation applied at test time only** — training always uses clean images; degradation is a test-time evaluation protocol, not a data augmentation strategy.
- **Forensic Trust Scorecard** — moves beyond binary accuracy by combining accuracy (0.4), IoU faithfulness (0.4), and SSIM stability (0.2) into a single deployability score.
