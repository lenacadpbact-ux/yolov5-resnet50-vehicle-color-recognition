################################################################################
# Confidence-Gated Vehicle Colour Recognition
# YOLOv5 Detection with ResNet-50 Fallback Classification
#
# Two-stage pipeline: YOLOv5 detects vehicles and gives an initial colour
# prediction; when its confidence falls below tau = 0.60, the cropped region
# is escalated to a ResNet-50 classifier for a fine-grained colour decision.
#
# Repository : https://github.com/lenacadpbact-ux/DeepLearning_Assignment_1
# Dataset    : https://universe.roboflow.com/final-project-jwpes/cars-color-recognition/dataset/2
# Author     : Arlen Balunan
################################################################################


################################################################################
# STEP 1 — Environment & Reproducibility Check
################################################################################
import torch, random, numpy as np, os, sys, platform
print("Python  :", sys.version)
print("Platform:", platform.platform())
print("PyTorch :", torch.version)
if torch.cuda.is_available():
    print("CUDA    :", torch.version.cuda)
    print("GPU     :", torch.cuda.get_device_name(0))
else:
    print("CUDA    : Not available (CPU mode)")

# Fix reproducibility seeds (helps with splits, anchor k-means, etc.)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


################################################################################
# STEP 2 — Install YOLOv5 + Dependencies
# Run this cell every time the Colab runtime disconnects/reconnects.
################################################################################
!git clone -q https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip -q install -r requirements.txt
!pip -q install torchmetrics==1.3.0 seaborn==0.13.2


################################################################################
# STEP 3 — Load the Dataset (YAML path) & Normalize Paths
################################################################################
# Load dataset YAML and normalize absolute paths
from pathlib import Path
import yaml

# Dataset YAML path
yaml_path = '/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/data.yaml'

# Load YAML
with open(yaml_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Make train/val/test paths absolute (YOLOv5 prefers absolute or correct relative)
base_dir = Path(yaml_path).parent
for split in ('train', 'val', 'test'):
    if split in cfg and cfg[split]:
        cfg[split] = str((base_dir / cfg[split]).resolve())

# Classes (order must match the label IDs in .txt files)
names = cfg.get('names')
print("Using dataset YAML:", yaml_path)
print("Classes:", names)

# Save a normalized copy used for training/val
norm_yaml = Path('/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/data_norm.yaml')
with open(norm_yaml, 'w') as f:
    yaml.safe_dump(cfg, f)
print("Normalized YAML written to:", norm_yaml)


################################################################################
# STEP 3.1 — Sanity Check (file existence + sample label pairs)
################################################################################
from pathlib import Path
import yaml

cfg = yaml.safe_load(open('/content/dataset/data_norm.yaml'))

def exists(p):
    if not p:
        return False
    if isinstance(p, list):
        return all(Path(x).exists() for x in p)
    return Path(p).exists()

print("train:", cfg.get('train'), "exists:", exists(cfg.get('train')))
print("val:  ", cfg.get('val'),   "exists:", exists(cfg.get('val')))
print("test: ", cfg.get('test'),  "exists:", exists(cfg.get('test')))

# Peek at a few images and their expected labels
def sample_check(imgs_dir):
    if not imgs_dir or not Path(imgs_dir).exists():
        print(f"[MISS] {imgs_dir}"); return
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    imgs = [p for p in Path(imgs_dir).rglob('*') if p.suffix.lower() in exts]
    print(f"Found {len(imgs)} images under {imgs_dir}")
    for im in imgs[:5]:
        lbl = Path(str(im).replace('/images/', '/labels/')).with_suffix('.txt')
        print(("V" if lbl.exists() else "X"), "label for", im.name, "->", lbl)

sample_check(cfg.get('train'))
sample_check(cfg.get('val'))

# --------------------------------------------------------------
# Logs
# --------------------------------------------------------------
# train: /content/dataset/train/images exists: True
# val:   /content/dataset/valid/images exists: True
# test:  /content/dataset/test/images exists: True
# Found 4461 images under /content/dataset/train/images
# Found 406 images under /content/dataset/valid/images


################################################################################
# STEP 3.2 — Re-normalize data.yaml
# NOTE: duplicates the normalization done in Step 3 — kept as-is from the
# original notebook, safe to remove if Step 3 already ran in this session.
################################################################################
from pathlib import Path
import yaml

yaml_path = "/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/data.yaml"
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

base_dir = Path(yaml_path).parent
for split in ("train", "val", "test"):
    if split in cfg and cfg[split]:
        cfg[split] = str((base_dir / cfg[split]).resolve())

norm_yaml = Path("/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/data_norm.yaml")
with open(norm_yaml, "w") as f:
    yaml.safe_dump(cfg, f)
print("Normalized YAML saved to:", norm_yaml)


################################################################################
# STEP 4 — Train YOLOv5 (50 Epochs) with Fixed Anchors
################################################################################
%cd /content/yolov5
!python train.py \
  --img 416 \
  --batch 4 \
  --epochs 50 \
  --data /content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/data_norm.yaml \
  --weights yolov5s.pt \
  --workers 2 \
  --cache none \
  --device cpu


################################################################################
# STEP 5 — Validate YOLO (mAP/PR) and Verify Anchors Retained
################################################################################
from pathlib import Path
import pandas as pd
import json, torch, os, sys, time

# ---- Paths ----
BEST_PT = "/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/runs/exp_t4_x640_b86/weights/best.pt"
RESULTS_DIR = "/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/runs/exp_t4_x640_b86/"
DATA_YAML = "/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/data_norm.yaml"
CONF_THRES = 0.74
IOU_THRES = 0.65

# Sanity checks
assert Path(BEST_PT).exists(), f"best.pt not found: {BEST_PT}"
assert Path(DATA_YAML).exists(), f"data_norm.yaml not found: {DATA_YAML}"

# Ensure yolov5 code is present
if not Path("/content/yolov5/val.py").exists():
    print("[INFO] YOLOv5 not found. Cloning...")
    !git clone -q https://github.com/ultralytics/yolov5 /content/yolov5
    %cd /content/yolov5
    !pip install -q -U pip
    !pip install -q -r requirements.txt
else:
    %cd /content/yolov5

# Keep logs clean / non-interactive
os.environ["WANDB_MODE"] = "disabled"

# ---------------------- Run Validation ----------------------
# Note: this val.py version does NOT support --plots. Removed.
PROJECT = "runs_val"
NAME = f"exp_val_conf{str(CONF_THRES).replace('.', '')}"
!python val.py \
  --weights "{BEST_PT}" \
  --data "{DATA_YAML}" \
  --task val \
  --imgsz 640 \
  --conf-thres {CONF_THRES} \
  --iou-thres {IOU_THRES} \
  --project {PROJECT} \
  --name {NAME} \
  --exist-ok

# Small wait to ensure files are flushed
time.sleep(1.0)

# ---------------------- Read Metrics ----------------------
search_roots = [Path(PROJECT), Path("runs/val")]  # support both layouts
val_dir = None
for root in search_roots:
    if root.exists():
        exps = sorted(root.glob("exp*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if exps:
            val_dir = exps[0]
            break

if val_dir is None:
    raise RuntimeError("No validation output directory found. Check val.py output above for errors.")

csv_path = val_dir / "results.csv"
if not csv_path.exists():
    # Some older versions write results.txt only; handle that case
    txt_path = val_dir / "results.txt"
    if txt_path.exists():
        print(f"[WARNING] results.csv not found; results.txt exists at {txt_path}")
    else:
        raise FileNotFoundError(f"Missing results.csv in {val_dir}.")
else:
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    print("\n====== VALIDATION SUMMARY ======")
    print(f"Summary from: {val_dir.resolve()}")
    print(f"Precision        : {last.get('precision', float('nan')):.4f}")
    print(f"Recall           : {last.get('recall', float('nan')):.4f}")
    print(f"mAP@0.5          : {last.get('mAP_0.5', float('nan')):.4f}")
    print(f"mAP@0.5:0.95     : {last.get('mAP_0.5:0.95', float('nan')):.4f}")
    print(f"Conf threshold   : {CONF_THRES}")
    print(f"IOU threshold    : {IOU_THRES}")
    print("\nKey artifacts directory:", val_dir.resolve())

# ---------------------- Extract Anchors ----------------------
anchors_used = None
ckpt = torch.load(BEST_PT, map_location="cpu")

# Try reading from Detect layer (most robust)
try:
    model_obj = ckpt.get("model", None)
    if model_obj is not None and hasattr(model_obj, "model") and len(model_obj.model) > 0:
        detect_layer = model_obj.model[-1]
        if hasattr(detect_layer, "anchors") and detect_layer.anchors is not None:
            anchors_used = detect_layer.anchors.detach().cpu().numpy().tolist()
except Exception as e:
    print("[INFO] Detect-layer anchor read failed:", e, file=sys.stderr)

# Fallback: YAML in checkpoint
if anchors_used is None:
    try:
        anchors_used = ckpt["model"].yaml.get("anchors", None)
    except Exception as e:
        print("[INFO] YAML anchor read failed:", e, file=sys.stderr)

print("\nAnchors found in best.pt:")
print(anchors_used)

# Save anchors next to the run folder on Drive
out_json = Path(RESULTS_DIR) / "anchors_used.json"
with open(out_json, "w") as f:
    json.dump(anchors_used, f, indent=2)
print(f"\nSaved anchors JSON to: {out_json.resolve()}")

# ==============================================================================
# Step 5 Results (log)
# ==============================================================================
#                  Class     Images  Instances          P          R      mAP50   mAP50-95
#                    all        406        406      0.983      0.973      0.982       0.98
#                  black        406         83          1      0.964      0.982      0.978
#                   blue        406         60      0.967      0.967      0.981      0.981
#                  green        406         64      0.984      0.969      0.983      0.982
#                    red        406         85          1      0.988      0.993      0.993
#                  white        406         62      0.984      0.968      0.982      0.975
#                 yellow        406         52      0.962      0.981      0.969      0.969
# Speed: 7.3ms pre-process, 678.6ms inference, 1.0ms NMS per image at shape (32, 3, 640, 640)
# Results saved to runs_val/exp_val_conf074


################################################################################
# STEP 6 — Build CNN Dataset (folder-per-colour) from YOLO Labels
################################################################################
from pathlib import Path
import yaml, cv2

# 1) Load dataset YAML and class names
DATA_YAML = "/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/data_norm.yaml"
cfg = yaml.safe_load(open(DATA_YAML, "r"))
names = cfg.get("names", [])
assert names, "No 'names' found in YAML."
cnn_classes = list(names)

# 2) Prepare output folders
cnn_root = Path("/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/cnn_data")
for split in ("train", "val", "test"):
    for c in cnn_classes:
        (cnn_root / split / c).mkdir(parents=True, exist_ok=True)

def to_list(x):
    if x is None or x == "":
        return []
    return x if isinstance(x, list) else [x]

def resolve_img_lbl_dirs(split_path: Path):
    """
    Accept either:
      - split_path = .../train/images  (common)
      - split_path = .../train         (has 'images' and 'labels' subfolders)
    Return (img_dir, lbl_dir)
    """
    p = Path(split_path)
    if p.name.lower() == "images":
        img_dir = p
        lbl_dir = p.parent / "labels"
    else:
        img_dir = p / "images"
        lbl_dir = p / "labels"
    return img_dir, lbl_dir

def yolo_boxes(lbl_path: Path, W: int, H: int):
    """Read YOLO txt label file -> list[(cls, x1,y1,x2,y2)] in pixel coords, clipped to image bounds."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    for ln in lbl_path.read_text().splitlines():
        ps = ln.strip().split()
        if len(ps) != 5:
            continue
        cls, cx, cy, w, h = int(ps[0]), float(ps[1]), float(ps[2]), float(ps[3]), float(ps[4])
        x1 = int(max(0, (cx - w / 2) * W)); y1 = int(max(0, (cy - h / 2) * H))
        x2 = int(min(W - 1, (cx + w / 2) * W)); y2 = int(min(H - 1, (cy + h / 2) * H))
        if x2 > x1 and y2 > y1:
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def make_crops(split_paths, split_name):
    """Create crops under /content/cnn_data/{split}/{class}/... from GT labels."""
    split_paths = to_list(split_paths)
    if not split_paths:
        print(f"[INFO] No '{split_name}' split in YAML; skipping.")
        return
    total = 0
    for sp in split_paths:
        img_dir, lbl_dir = resolve_img_lbl_dirs(Path(sp))
        if not img_dir.exists():
            print(f"[WARN] Missing images dir: {img_dir} (skipping)")
            continue
        imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        if not imgs:
            print(f"[WARN] No images found under {img_dir}")
            continue
        counters = {c: 0 for c in cnn_classes}
        for im in imgs:
            I = cv2.imread(str(im))
            if I is None:
                continue
            H, W = I.shape[:2]
            rel = im.relative_to(img_dir).with_suffix(".txt")
            lbl = lbl_dir / rel
            for (cls, x1, y1, x2, y2) in yolo_boxes(lbl, W, H):
                if cls < 0 or cls >= len(names):
                    continue
                class_name = names[cls]
                crop = I[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (128, 128))
                out_path = cnn_root / split_name / class_name / f"{im.stem}_{counters[class_name]}.jpg"
                cv2.imwrite(str(out_path), crop)
                counters[class_name] += 1
                total += 1
    print(f"[{split_name}] total crops saved: {total}")

# 3) Build crops for all splits present in YAML
make_crops(cfg.get("train"), "train")
make_crops(cfg.get("val"), "val")     # NOTE: YAML key is 'val' even if folder is 'valid'
make_crops(cfg.get("test"), "test")
print("CNN crop dataset created under:", cnn_root)

# --------------------------------------------------
# Logs
# --------------------------------------------------
# [train] total crops saved: 4461
# [val] total crops saved: 406
# [test] total crops saved: 210


################################################################################
# STEP 7 — Train ResNet-50 (50 Epochs) on the CNN Dataset
################################################################################
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

# 1. Force GPU check
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")

# 2. Path & params
cnn_root = Path("/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/cnn_data")
NUM_CLASSES = len(cnn_classes)
EPOCHS = 50
BATCH = 128

# 3. Data loading
train_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
eval_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
dsets = {
    "train": datasets.ImageFolder(str(cnn_root / "train"), transform=train_tfms),
    "val":   datasets.ImageFolder(str(cnn_root / "val"), transform=eval_tfms)
}
dls = {k: DataLoader(v, batch_size=BATCH, shuffle=(k == "train"),
                     num_workers=4, pin_memory=True, persistent_workers=True)
       for k, v in dsets.items()}

# 4. Model setup
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
resnet = resnet.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)

# 5. Training loop
best_acc = 0.0
for ep in range(1, EPOCHS + 1):
    resnet.train()
    for X, y in dls["train"]:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = resnet(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    # Validation
    resnet.eval()
    va = vn = 0
    with torch.no_grad():
        for X, y in dls["val"]:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = resnet(X)
            va += (logits.argmax(1) == y).sum().item()
            vn += X.size(0)

    val_acc = va / vn
    print(f"Epoch {ep:02d} | Val Acc: {val_acc:.4f}")

# --------------------------------------------------------------
# Logs (abbreviated)
# --------------------------------------------------------------
# Training on: cuda:0
# Epoch 01 | Val Acc: 0.9163
# ...
# Epoch 44 | Val Acc: 0.9803   (peak, per Appendix log = 98.28%)
# Epoch 50 | Val Acc: 0.9778


################################################################################
# STEP 8 — Unified Fusion Inference (Single & Batch)
################################################################################
from pathlib import Path
import torchvision.transforms as T
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes as scale_coords
from utils.augmentations import letterbox
import cv2, numpy as np, torch, matplotlib.pyplot as plt
from datetime import datetime

# 1. Configuration
cnn_classes = ['black', 'blue', 'green', 'red', 'white', 'yellow']
names = cnn_classes
FUSE_THRESHOLD = 0.60
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
DEMO_SINGLE_IMAGE = False
DEMO_SAVE_PATH = f"runs_yolov5/fused/inference_demo_{now_str}.jpg"

# 2. Load models
weights_path = "/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/runs/exp_t4_x640_b86/weights/best.pt"
detector = DetectMultiBackend(weights_path, device=device)
detector.model.float().eval()
resnet.eval()
out_dir = Path("runs_yolov5/fused")
out_dir.mkdir(parents=True, exist_ok=True)

# 3. Fusion logic (fixed size filters + debug prints)
def fuse_color(I_bgr, xyxy_conf_cls):
    x1, y1, x2, y2, conf, cls = xyxy_conf_cls

    box_w = x2 - x1
    box_h = y2 - y1
    img_h, img_w = I_bgr.shape[:2]

    # Too small — ignore noise
    if box_w < 0.02 * img_w or box_h < 0.02 * img_h:
        print(f"  [SKIP-small]  box=({box_w:.0f}x{box_h:.0f})  img=({img_w}x{img_h})")
        return None, None, None

    yolo_label = names[int(cls)]

    # Low-confidence YOLO -> defer to ResNet
    if yolo_label in cnn_classes and float(conf) < FUSE_THRESHOLD:
        crop = I_bgr[int(y1):int(y2), int(x1):int(x2)]
        if crop.size > 0:
            crop = cv2.cvtColor(cv2.resize(crop, (128, 128)), cv2.COLOR_BGR2RGB)
            tens = T.ToTensor()(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(resnet(tens), dim=1).squeeze().cpu().numpy()
            cid = int(np.argmax(probs))
            return cnn_classes[cid], float(probs[cid]), "ResNet"

    return yolo_label, float(conf), "YOLO"

# 4. Drawing
def draw_box(img, x1, y1, x2, y2, label, score, src):
    color = (0, 255, 0) if src == "YOLO" else (72, 180, 255)
    thickness = 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    text = f"{label} {score:.2f} ({src})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)

    if y1 - th - 8 > 0:
        b_y1, b_y2, t_y = y1 - th - 8, y1, y1 - 4
    else:
        b_y1, b_y2, t_y = y1, y1 + th + 8, y1 + th + 4

    cv2.rectangle(img, (x1, b_y1), (x1 + tw + 8, b_y2), color, -1)
    cv2.putText(img, text, (x1 + 4, t_y), font, font_scale,
                (0, 0, 0), font_thick, cv2.LINE_AA)

# 5. Main execution
if DEMO_SINGLE_IMAGE:
    # -- Single image mode --
    try:
        from google.colab import files
        print("Upload one test image (JPG/PNG)...")
        uploaded = files.upload()
        demo_img_path = list(uploaded.keys())[0]
    except Exception:
        demo_img_path = "YOUR_IMAGE.jpg"

    I0 = cv2.imread(str(demo_img_path))
    L = letterbox(I0, 640, stride=detector.stride, auto=True)[0]
    im_np = L[:, :, ::-1].transpose(2, 0, 1).copy()
    im = torch.from_numpy(im_np).to(device).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        pred = detector(im)
        pred = non_max_suppression(pred, 0.25, 0.45, max_det=100)[0]

    annotated = I0.copy()
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], I0.shape).round()
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            label, score, src = fuse_color(I0, [x1, y1, x2, y2, conf.item(), cls.item()])
            if label is not None:
                draw_box(annotated, x1, y1, x2, y2, label, score, src)
                print(f"-> {label} {score:.2f} ({src})")
    else:
        print("No detections")

    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(I0, cv2.COLOR_BGR2RGB))
    plt.title("Original Input Environment", fontsize=14, pad=15)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("Hierarchical Fusion Output", fontsize=14, pad=15)
    plt.axis('off')
    plt.tight_layout(pad=3.0)
    plt.savefig(DEMO_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')

else:
    # -- Batch mode --
    test_img_dir = Path("/content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/dataset/test/images")
    print("Test dir:", test_img_dir)
    print("Exists  :", test_img_dir.exists())

    samples = sorted([p for p in test_img_dir.glob("*")
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png")])[:20]
    print(f"Processing {len(samples)} images...\n")

    for im_p in samples:
        I0 = cv2.imread(str(im_p))
        if I0 is None:
            print(f"Cannot read: {im_p.name}")
            continue

        L = letterbox(I0, 640, stride=detector.stride, auto=True)[0]
        im = torch.from_numpy(
            L[:, :, ::-1].transpose(2, 0, 1).copy()
        ).to(device).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = detector(im)
            pred = non_max_suppression(pred, 0.25, 0.45, max_det=100)[0]

        annotated = I0.copy()

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], I0.shape).round()
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = map(int, xyxy)
                label, score, src = fuse_color(
                    I0, [x1, y1, x2, y2, conf.item(), cls.item()])
                if label is not None:
                    draw_box(annotated, x1, y1, x2, y2, label, score, src)
                    print(f"  {im_p.name} -> {label} {score:.2f} ({src})")
        else:
            print(f"  {im_p.name} — no detection")

        out_path = out_dir / f"{im_p.stem}_fused_{now_str}.jpg"
        cv2.imwrite(str(out_path), annotated)

    print(f"\nAll results saved to: {out_dir}")

# ----------------------------------------------------------------
# Logs after successful annotations (abbreviated)
# ----------------------------------------------------------------
# Fusing layers...
# Model summary: 157 layers, 7026307 parameters, 0 gradients, 15.8 GFLOPs
# Test dir: /content/drive/MyDrive/ASSIGNMENT/Assignment_Project_1/dataset/test/images
# Exists  : True
# Processing 20 images...
#   auto-3309967_640_jpg....jpg -> black 1.00 (ResNet)
# All results saved to: runs_yolov5/fused


################################################################################
# STEP 9 — MATLAB R2025b: ResNet-50 Convergence Plot
################################################################################
% ResNet-50 Convergence Analysis — Updated Run
epochs = 1:50;

val_acc = [0.9039, 0.9606, 0.9631, 0.9655, 0.9655, 0.9680, 0.9704, 0.9729, ...
           0.9729, 0.9729, 0.9729, 0.9729, 0.9729, 0.9754, 0.9729, 0.9729, ...
           0.9729, 0.9729, 0.9754, 0.9729, 0.9754, 0.9754, 0.9754, 0.9754, ...
           0.9704, 0.9754, 0.9754, 0.9754, 0.9729, 0.9754, 0.9754, 0.9754, ...
           0.9754, 0.9704, 0.9729, 0.9754, 0.9778, 0.9754, 0.9729, 0.9754, ...
           0.9778, 0.9729, 0.9729, 0.9828, 0.9803, 0.9778, 0.9778, 0.9778, ...
           0.9778, 0.9778] * 100;

% -- Square figure --
figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 5.5 5.5]);

plot(epochs, val_acc, '-o', 'LineWidth', 1.5, 'MarkerSize', 4, ...
    'Color', [0.12 0.47 0.71]);
grid on;
hold on;

% Peak marker — dark green
[peak_acc, peak_idx] = max(val_acc);
plot(epochs(peak_idx), peak_acc, '*', 'MarkerSize', 10, 'LineWidth', 2, ...
    'Color', [0.1 0.5 0.1]);

% -- Text inside the box — top left area --
text(3, 99.5, sprintf('Peak: %.2f%% (Epoch %d)', peak_acc, epochs(peak_idx)), ...
    'Color', [0.1 0.5 0.1], 'FontSize', 10, 'FontWeight', 'bold');

% Convergence annotation — bottom right inside box
text(15, 90.8, 'Convergence achieved at Epoch 8-9', ...
    'Color', [0.5 0.5 0.5], 'FontSize', 9, 'FontAngle', 'italic');

% Peak dashed line — dark green
yline(peak_acc, '--', 'LineWidth', 0.8, 'Color', [0.1 0.5 0.1]);

% -- Labels --
xlabel('Training Epoch', 'FontSize', 12);
ylabel('Validation Accuracy (%)', 'FontSize', 12);
title('ResNet-50 Convergence Analysis (50 Epochs)', 'FontSize', 13, 'FontWeight', 'bold');
ylim([90 100]);
xlim([0 51]);

hold off;

% -- Save --
exportgraphics(gcf, 'resnet50_convergence_square.png', 'Resolution', 300);
