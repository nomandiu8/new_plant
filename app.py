"""
PlantSeg Decision-Support Demo — Gradio deployment app.

This is a live, public-facing wrapper around the exact inference pipeline described in
Section 3.7 / 4.5 of the paper "Faithfulness-Gated Decision Support for In-the-Wild Plant
Disease Diagnosis". It chains the best classifier (ConvNeXtV2-Tiny) and the best
convolutional segmenter (DeepLabV3+ / EfficientNet-B3) into a single tool that returns,
for any uploaded leaf/fruit image:

  1. a predicted disease class,
  2. a lesion-area severity estimate (from the predicted segmentation mask),
  3. a ground-truth-free "confidence flag" that checks whether the classifier's own
     Grad-CAM attention falls inside its own predicted lesion region.

The model-construction and inference code below is copied verbatim (not reimplemented)
from the already-executed and paper-verified notebook
`PlantSeg_Stage5_Decision_Support_Demo.ipynb`, so the numbers this app produces are
consistent with what is reported in the manuscript.

IMPORTANT — do not swap in a transformer segmenter (e.g. SegFormer-B2) here. Section 4 of
the paper found attention-rollout attribution on the transformer does not reliably
localise the lesion (CAM-GT IoU 0.067 vs 0.434 for the CNN pipeline), so a HIGH confidence
flag from a transformer-based version of this app would not carry the same meaning. This
asymmetry is itself one of the paper's findings, and it is why this app deliberately uses
only the CNN classifier + CNN segmenter pair whose faithfulness was actually validated.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import segmentation_models_pytorch as smp
import gradio as gr
import spaces
from PIL import Image
from pytorch_grad_cam import GradCAM
from huggingface_hub import hf_hub_download
import os

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
HERE = Path(__file__).parent

# --- Model download from Hugging Face Hub ---
# Set HF_MODEL_REPO env var to your model repo (e.g., "username/plantseg-models")
# Models will be auto-downloaded on first run.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")

def get_model_path(filename):
    """Download model from HF Hub if HF_MODEL_REPO is set, else look locally."""
    local_path = HERE / filename
    if local_path.exists():
        return str(local_path)
    if HF_MODEL_REPO:
        print(f"Downloading {filename} from {HF_MODEL_REPO}...")
        return hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename)
    raise FileNotFoundError(
        f"{filename} not found locally and HF_MODEL_REPO is not set. "
        f"Either place the file in {HERE} or set the HF_MODEL_REPO environment variable."
    )

CLS_CKPT = get_model_path("ConvNeXtV2Tiny_best.pt")
SEG_CKPT = get_model_path("DeepLabV3Plus_efficientnet-b3.pt")
CLASS_NAMES_FILE = HERE / "class_names.json"

IMG_SIZE_CLS = 384   # ConvNeXtV2-Tiny trained with IMG_SIZE_CNN=384
IMG_SIZE_SEG = 384   # DeepLabV3+ using same resolution
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_NAMES_FILE) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)


# ============================================================================
# MODEL DEFINITIONS (identical to the Stage-1 / Stage-2 trainer notebooks and to
# PlantSeg_Stage5_Decision_Support_Demo.ipynb — do not change the architecture here
# without re-validating against the paper's reported numbers)
# ============================================================================
def build_classifier(num_classes):
    return timm.create_model("convnextv2_tiny", pretrained=False, num_classes=num_classes)


class SegNet(nn.Module):
    """DeepLabV3+ / EfficientNet-B3, binary lesion-vs-background, unified full-res logits."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.net = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        return self.net(x)


def load_state(model, ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model", "best_state"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    raw = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(raw, strict=False)
    return model


print("Loading models...")
clf = build_classifier(NUM_CLASSES)
clf = load_state(clf, CLS_CKPT).to(device).eval()

seg = SegNet(num_classes=2)
seg = load_state(seg, SEG_CKPT).to(device).eval()
print(f"Both models loaded on {device}.")

cls_tf = T.Compose([T.Resize((IMG_SIZE_CLS, IMG_SIZE_CLS)), T.ToTensor(), T.Normalize(MEAN, STD)])
seg_tf = T.Compose([T.Resize((IMG_SIZE_SEG, IMG_SIZE_SEG)), T.ToTensor(), T.Normalize(MEAN, STD)])

# Grad-CAM on the classifier's last ConvNeXtV2 stage (matches Stage-1's own `tgt()` mapping)
cam_extractor = GradCAM(model=clf, target_layers=[clf.stages[-1]])


# ============================================================================
# CORE PIPELINE: image -> class + severity + faithfulness-gated confidence
# (verbatim logic from PlantSeg_Stage5_Decision_Support_Demo.ipynb, cell 6)
# ============================================================================
def diagnose_image(pil_img: Image.Image):
    img = pil_img.convert("RGB")

    # 1) Classification
    x_cls = cls_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = clf(x_cls)
        probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    cls_confidence = float(probs[pred_idx])

    # 2) Lesion segmentation -> severity (lesion-area ratio, matching Section 3.4 of the paper)
    x_seg = seg_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_logits = seg(x_seg)
        pred_mask = seg_logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # (H,W) in {0,1}
    severity = float(pred_mask.mean())

    # 3) Grad-CAM for the predicted class, at the classifier's own input resolution
    cam = cam_extractor(input_tensor=x_cls, targets=None)[0]  # (H,W) in [0,1], size IMG_SIZE_CLS

    # 4) Faithfulness-gated confidence flag (deployment-time proxy for Section 4's CAM-GT check)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE_SEG, IMG_SIZE_SEG))
    cam_rs = np.asarray(cam_img).astype(np.float32) / 255.0
    tau = max(severity, 1e-3)
    thresh = np.quantile(cam_rs, 1 - tau)
    cam_bin = (cam_rs >= thresh).astype(np.uint8)
    inter = np.logical_and(cam_bin, pred_mask).sum()
    union = np.logical_or(cam_bin, pred_mask).sum()
    cam_pred_iou = float(inter / union) if union > 0 else 0.0
    peak_yx = np.unravel_index(np.argmax(cam_rs), cam_rs.shape)
    pointing_hit = bool(pred_mask[peak_yx] == 1)
    confidence_flag = "HIGH" if pointing_hit else "LOW"

    return dict(
        pred_class=pred_class,
        cls_confidence=cls_confidence,
        severity=severity,
        cam_pred_iou=cam_pred_iou,
        pointing_hit=pointing_hit,
        confidence_flag=confidence_flag,
        img=img.resize((IMG_SIZE_SEG, IMG_SIZE_SEG)),
        pred_mask=pred_mask,
        cam=cam_rs,
    )


# ============================================================================
# VISUALISATION
# ============================================================================
def make_lesion_overlay(base_img: Image.Image, mask: np.ndarray, alpha: float = 0.5):
    """Lesion mask: red fill + bright green contour border for clarity."""
    import cv2
    base = np.asarray(base_img).astype(np.float32) / 255.0
    h, w = base.shape[:2]
    mask_rs = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h))) / 255.0
    red_overlay = np.zeros_like(base)
    red_overlay[..., 0] = 1.0
    blended = np.where(mask_rs[..., None] > 0.5,
                       (1 - alpha) * base + alpha * red_overlay, base)
    mask_u8 = (mask_rs > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blended_u8 = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    cv2.drawContours(blended_u8, contours, -1, (0, 255, 80), 2)
    return Image.fromarray(blended_u8)


def make_cam_overlay(base_img: Image.Image, cam: np.ndarray, alpha: float = 0.65):
    """Grad-CAM: Gaussian-smoothed heatmap with turbo colormap for vivid visualization."""
    import matplotlib.cm as cm
    from scipy.ndimage import gaussian_filter
    base = np.asarray(base_img).astype(np.float32) / 255.0
    h, w = base.shape[:2]
    cam_rs = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h),
                      resample=Image.BILINEAR)) / 255.0
    cam_smooth = gaussian_filter(cam_rs, sigma=8)
    cam_norm = cam_smooth - cam_smooth.min()
    if cam_norm.max() > 0:
        cam_norm /= cam_norm.max()
    colormap = cm.get_cmap("turbo")
    colored = colormap(cam_norm)[..., :3]
    heat_mask = (cam_norm > 0.2).astype(np.float32)
    blended = (1 - alpha * heat_mask[..., None]) * base + (alpha * heat_mask[..., None]) * colored
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


@spaces.GPU
def diagnose(image):
    if image is None:
        return None, None, None, "Upload an image first."

    res = diagnose_image(image)

    lesion_overlay = make_lesion_overlay(res["img"], res["pred_mask"])
    cam_overlay = make_cam_overlay(res["img"], res["cam"])

    flag = res["confidence_flag"]
    flag_note = (
        "High confidence: the classifier's attention falls inside its own predicted "
        "lesion region — the prediction is self-consistent."
        if flag == "HIGH"
        else "Low confidence: the classifier's attention does NOT fall inside its own "
        "predicted lesion region — treat this prediction as unreliable and consider "
        "manual review."
    )

    summary = f"""
### Predicted disease: **{res['pred_class']}**

| Metric | Value |
|---|---|
| Classification confidence | {res['cls_confidence']*100:.1f}% |
| Predicted lesion severity | {res['severity']*100:.1f}% of leaf/fruit area |
| CAM–predicted-mask IoU | {res['cam_pred_iou']:.3f} |
| **Confidence flag** | **{flag}** |

{flag_note}

*Ground-truth-free at inference time — this flag is computed purely from the
classifier's own Grad-CAM against the segmenter's own predicted mask, with no access to
expert annotations. On the full PlantSeg held-out test set (n=1,561), images flagged
HIGH were classified correctly 72.4% of the time vs. 62.1% for LOW (Fisher's exact
p<0.0001) — see Table 12 of the paper.*
"""
    return res["img"], lesion_overlay, cam_overlay, summary


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
DESCRIPTION = """
# 🌿 PlantSeg Faithfulness-Gated Decision Support

Upload a photo of a plant leaf or fruit. This tool runs the exact classifier +
segmenter pipeline described in the paper *"Faithfulness-Gated Decision Support for
In-the-Wild Plant Disease Diagnosis"* and returns a disease class, a lesion-area
severity estimate, and a **confidence flag** — computed entirely at inference time,
with no access to ground truth.

**Scope note.** This pipeline is validated only for the convolutional
classifier + convolutional segmenter pair used here (ConvNeXtV2-Tiny +
DeepLabV3+/EfficientNet-B3). The paper found that Grad-CAM attention is faithful for
convolutional architectures but not for the transformer segmenter tested (SegFormer-B2),
so the confidence flag should not be assumed valid for other model combinations without
re-validation.
"""

css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
h1 {
    text-align: center;
    margin-bottom: 0.5em;
}
"""

with gr.Blocks(title="PlantSeg Decision Support", css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload leaf/fruit image", height=400)
    with gr.Row():
        run_btn = gr.Button("🔬 Diagnose", variant="primary", size="lg")
    with gr.Row():
        out_input = gr.Image(label="Input (resized)")
        out_lesion = gr.Image(label="Predicted lesion (severity overlay)")
        out_cam = gr.Image(label="Grad-CAM (classifier attention)")
    out_summary = gr.Markdown()

    run_btn.click(fn=diagnose, inputs=inp, outputs=[out_input, out_lesion, out_cam, out_summary])
    inp.change(fn=diagnose, inputs=inp, outputs=[out_input, out_lesion, out_cam, out_summary])

if __name__ == "__main__":
    demo.launch()
