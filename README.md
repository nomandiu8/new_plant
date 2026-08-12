---
title: PlantSeg Decision Support
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.31.0"
app_file: app.py
pinned: true
license: mit
---

# 🌿 PlantSeg Faithfulness-Gated Decision Support

A live deployment of the inference pipeline from the paper:

> **"Faithfulness-Gated Decision Support for In-the-Wild Plant Disease Diagnosis"**

## Architecture

This repository contains two parts that work together, not two alternative deployments:

1. **`app.py` (this directory)** — the actual model-serving backend: loads the trained
   ConvNeXtV2-Tiny classifier and DeepLabV3+/EfficientNet-B3 segmenter, runs Grad-CAM, and
   computes the faithfulness-gated confidence flag. Deployed as a Hugging Face Space
   (`sdk: gradio` in the frontmatter above) at `nomandiu9/diseases_prediction`, exposed via
   its `/diagnose` Gradio API endpoint.
2. **`vercel-frontend/`** — a custom-styled public web UI hosted on Vercel at
   **[plant-seg.vercel.app](https://plant-seg.vercel.app/)**. Its `api/predict.py` is a thin
   proxy: it forwards the uploaded image to the Hugging Face Space above via `gradio_client`,
   parses the returned markdown summary into structured fields (disease class, severity,
   CAM–mask IoU, confidence flag), and the static `index.html` renders them.

**plant-seg.vercel.app is the link to use** when citing or demoing this deployment — it's the
polished public entry point. The Hugging Face Space is the model server behind it and isn't
meant to be visited directly, though it can be run standalone (see "How to Run Locally" below)
for testing or development.

## What it does

Upload a photo of a plant leaf or fruit, and the tool returns:

1. **Predicted disease class** — from a ConvNeXtV2-Tiny classifier
2. **Lesion severity estimate** — percentage of leaf/fruit area affected (from DeepLabV3+/EfficientNet-B3 segmenter)
3. **Confidence flag** — a ground-truth-free self-consistency check: does the classifier's own Grad-CAM attention fall inside its predicted lesion region?

## Models

| Component | Architecture | Purpose |
|---|---|---|
| Classifier | ConvNeXtV2-Tiny | Disease class prediction |
| Segmenter | DeepLabV3+ / EfficientNet-B3 | Binary lesion mask |
| Attribution | Grad-CAM | Classifier attention map |

## Important Note

This pipeline uses **only** the CNN classifier + CNN segmenter pair. The paper found that transformer-based segmenters (e.g., SegFormer-B2) do not produce faithful attention maps for this task (CAM-GT IoU 0.067 vs 0.434), so the confidence flag is validated only for this specific model combination.

## How to Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Place your model weights (`ConvNeXtV2Tiny_best.pt` and `DeepLabV3Plus_efficientnet-b3.pt`) in the same directory as `app.py`, or set the `HF_MODEL_REPO` environment variable to auto-download from Hugging Face Hub.
