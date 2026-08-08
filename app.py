"""
Confidence-Gated Vehicle Colour Recognition — Gradio Demo
Two-stage pipeline: YOLOv5 (detection + fast colour prediction) with a
ResNet-50 fallback classifier for low-confidence cases.

Deploy this on Hugging Face Spaces (free, no server management needed):
1. Create a new Space at https://huggingface.co/new-space
   - SDK: Gradio
   - Hardware: CPU basic (free tier is fine)
2. Upload this file as app.py
3. Upload requirements.txt (included below)
4. Upload your trained weights:
   - best.pt              (YOLOv5 weights)
   - resnet50_color_best.pt   (ResNet-50 weights)
5. The Space will auto-build and give you a public URL to share.

Update the WEIGHTS_* paths below if your filenames differ.
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import urllib.request

# ── Configuration ────────────────────────────────────────────────────────────
CNN_CLASSES = ['black', 'blue', 'green', 'red', 'white', 'yellow']
FUSE_THRESHOLD = 0.60
WEIGHTS_YOLO = "best.pt"
WEIGHTS_RESNET = "resnet50_color_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Auto-download weights from Hugging Face Hub if not already present ──────
HF_REPO = "lenacadpbact/vehicle-colour-recognition-weights"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

def ensure_weights():
    for fname in (WEIGHTS_YOLO, WEIGHTS_RESNET):
        if not os.path.exists(fname):
            url = f"{HF_BASE_URL}/{fname}"
            print(f"Downloading {fname} from {url} ...")
            urllib.request.urlretrieve(url, fname)
            print(f"Downloaded {fname} ({os.path.getsize(fname) / 1e6:.1f} MB)")
        else:
            print(f"{fname} already present locally, skipping download.")

ensure_weights()

# ── Load YOLOv5 (via torch.hub — required for YOLOv5-format weights;
# the newer `ultralytics` YOLO() class only supports YOLOv8+ checkpoints) ───
yolo_model = torch.hub.load("ultralytics/yolov5", "custom", path=WEIGHTS_YOLO, force_reload=False, trust_repo=True)
yolo_model.to(DEVICE).eval()

# ── Load ResNet-50 fallback classifier ───────────────────────────────────────
resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, len(CNN_CLASSES))
checkpoint = torch.load(WEIGHTS_RESNET, map_location=DEVICE)
state_dict = checkpoint.get("state_dict", checkpoint)
resnet.load_state_dict(state_dict)
resnet.to(DEVICE).eval()

eval_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ── Fusion inference ──────────────────────────────────────────────────────────
def predict(image: Image.Image):
    if image is None:
        return None, "Please upload an image."

    img_np = np.array(image.convert("RGB"))
    results = yolo_model(img_np)
    detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls

    if len(detections) == 0:
        return image, "No vehicle detected. Try a clearer photo."

    annotated = img_np.copy()
    labels_out = []

    for x1, y1, x2, y2, conf, cls in detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        yolo_label = CNN_CLASSES[int(cls)] if int(cls) < len(CNN_CLASSES) else "unknown"

        if conf >= FUSE_THRESHOLD:
            label, score, source = yolo_label, float(conf), "YOLOv5"
        else:
            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_pil = Image.fromarray(crop)
            tensor = eval_tfms(crop_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(resnet(tensor), dim=1).squeeze().cpu().numpy()
            idx = int(np.argmax(probs))
            label, score, source = CNN_CLASSES[idx], float(probs[idx]), "ResNet-50 (fallback)"

        colour_box = (0, 255, 0) if source == "YOLOv5" else (255, 140, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour_box, 3)
        text = f"{label} {score:.2f} ({source})"
        cv2.putText(annotated, text, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour_box, 2)
        labels_out.append(text)

    summary = "\n".join(labels_out) if labels_out else "No confident predictions."
    return Image.fromarray(annotated), summary


# ── Gradio interface ──────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a car photo"),
    outputs=[
        gr.Image(type="pil", label="Detected & Classified"),
        gr.Textbox(label="Prediction details"),
    ],
    title="🚗 Confidence-Gated Vehicle Colour Recognition",
    description=(
        "Two-stage pipeline: YOLOv5 detects the vehicle and predicts colour. "
        "If YOLOv5's confidence is below 0.60, a ResNet-50 fallback classifier "
        "steps in for a more careful, fine-grained colour prediction. "
        "Trained on 5,077 vehicle images, achieving 98.28% peak validation accuracy. "
        "Upload any car photo to try it out!"
    ),
    examples=None,  # add example image paths here once uploaded to the Space
    allow_flagging="never",
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)

