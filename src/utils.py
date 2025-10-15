import os, json, random, glob
import numpy as np
import torch

from torchvision.transforms import v2
from torchvision.io import read_image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score 


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def best_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def eval_metrics(y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    auc  = float("nan")
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return {"acc":round(acc, 4), "f1":round(f1, 4), "prec":round(prec, 4), "rec":round(rec, 4), "auc":round(auc, 4)}

def save_confusion_matrix(y_true, y_pred, path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_sample_preview(data_dir, out_path, img_size=28):
    candidates = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    if not candidates:
        print("[warn] No .jpg files found for preview.")
        return
    import random as _r
    img_path = _r.choice(candidates)
    img_raw = read_image(img_path)
    transform = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.RandomResizedCrop(size=(img_size, img_size), scale=(1.0, 1.0)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    img_t = transform(img_raw)
    if torch.is_floating_point(img_t) and float(img_t.min()) < 0:
        img_t = img_t - img_t.min()
        img_t = img_t / (img_t.max() + 1e-6)
    def to_hwc_uint8(t):
        if t.dtype.is_floating_point:
            t = (t.clamp(0, 1) * 255.0).to(torch.uint8)
        return t.permute(1, 2, 0).cpu().numpy()
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(to_hwc_uint8(img_raw)); axs[0].set_title("Original"); axs[0].axis("off")
    axs[1].imshow(to_hwc_uint8(img_t));   axs[1].set_title("Transformed"); axs[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

