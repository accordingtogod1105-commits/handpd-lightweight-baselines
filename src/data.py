# data.py
from __future__ import annotations

import os
import glob
import shutil
import zipfile
import urllib.request
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from utils import ensure_dir

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

HANDPD_URLS = {
    "Meander": "https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/Meander_HandPD.zip",
    "Spiral":  "https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/Spiral_HandPD.zip",
}

ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _remove_macos_cruft(root: str) -> None:
    """Remove __MACOSX folders and AppleDouble files (._*) & .DS_Store."""
    for dirpath, dirnames, filenames in os.walk(root):
        # remove __MACOSX folders
        if "__MACOSX" in dirnames:
            try:
                shutil.rmtree(os.path.join(dirpath, "__MACOSX"), ignore_errors=True)
            except Exception:
                pass
        # remove AppleDouble and .DS_Store
        for fn in list(filenames):
            if fn.startswith("._") or fn == ".DS_Store":
                fp = os.path.join(dirpath, fn)
                try:
                    os.remove(fp)
                except FileNotFoundError:
                    pass


def _has_any_image(root: str) -> bool:
    """Check if root contains at least one allowed image file."""
    for ext in ALLOWED_EXTS:
        if glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True):
            return True
    return False


def _is_valid_image(path: str) -> bool:
    """Filter for torchvision.datasets.ImageFolder(is_valid_file=...)."""
    bn = os.path.basename(path)
    if "__MACOSX" in path or bn.startswith("._") or bn == ".DS_Store":
        return False
    return bn.lower().endswith(ALLOWED_EXTS)


def _default_num_workers() -> int:
    # cross-platform sensible default
    if os.name == "nt":
        return 0
    cpu = os.cpu_count() or 2
    return min(4, cpu)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def download_handpd(data_root: str = "./data/Handpd") -> None:
    """Download and unzip HandPD datasets; clean macOS artifacts."""
    ensure_dir(data_root)
    for name, url in HANDPD_URLS.items():
        zip_path = os.path.join(data_root, f"{name}_HandPD.zip")
        out_dir = os.path.join(data_root, f"{name}_HandPD")

        # Skip if dataset directory already exists and has images
        if os.path.exists(out_dir) and os.path.isdir(out_dir) and _has_any_image(out_dir):
            print(f"[skip] {name} found at {out_dir}")
            continue

        print(f"[download] {name}: {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except Exception:
                    pass
            raise RuntimeError(f"Failed to download {name} from {url}: {e}") from e

        print(f"[unzip] -> {out_dir}")
        ensure_dir(out_dir)
        try:
            if not zipfile.is_zipfile(zip_path):
                raise RuntimeError(f"Downloaded file is not a valid zip: {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
        finally:
            # Always try to remove the zip once we've attempted extraction
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except Exception:
                    pass

        # Clean macOS artifacts and verify contents
        _remove_macos_cruft(out_dir)
        if not _has_any_image(out_dir):
            raise RuntimeError(
                f"No images found in {out_dir} after extraction. "
                f"Please verify the archive structure or re-run with --download."
            )

    print("[done] Dataset ready.")


def _stratified_indices(targets: List[int], test_size: float, seed: int) -> Tuple[List[int], List[int]]:
    """Create stratified train/val indices per class; fallback to simple split if needed."""
    import random
    random.seed(seed)

    by_cls = {}
    for idx, y in enumerate(targets):
        by_cls.setdefault(int(y), []).append(idx)

    # Require at least 2 classes for stratification
    if len(by_cls) < 2:
        n = len(targets)
        all_idx = list(range(n))
        random.shuffle(all_idx)
        n_val = int(round(test_size * n))
        return all_idx[n_val:], all_idx[:n_val]

    train_idx, val_idx = [], []
    for cls, idxs in by_cls.items():
        if len(idxs) == 0:
            continue
        random.shuffle(idxs)
        n_val = max(1, int(round(test_size * len(idxs))))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    # If one split is empty (rare edge when tiny data), fallback to simple split
    if len(train_idx) == 0 or len(val_idx) == 0:
        all_idx = list(range(len(targets)))
        random.shuffle(all_idx)
        n_val = int(round(test_size * len(all_idx)))
        return all_idx[n_val:], all_idx[:n_val]

    return train_idx, val_idx


def load_dataloader(
    data_dir: str,
    img_size: int = 28,
    batch_size: int = 32,
    test_size: float = 0.3,
    seed: int = 42,
    num_workers: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build stratified train/val dataloaders from an ImageFolder dataset.
    - data_dir should contain class subfolders (e.g., Control/Patients).
    - Filters macOS artifacts and only loads allowed image extensions.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"[error] Data directory not found: {data_dir}")

    to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    transform = nn.Sequential(
        v2.Grayscale(num_output_channels=1),
        v2.RandomResizedCrop(size=(img_size, img_size), scale=(1.0, 1.0)),
        to_tensor,
    )

    dataset = ImageFolder(root=data_dir, transform=transform, is_valid_file=_is_valid_image)

    if len(dataset) == 0:
        # try cleaning and re-checking (in case user extracted manually)
        _remove_macos_cruft(data_dir)
        dataset = ImageFolder(root=data_dir, transform=transform, is_valid_file=_is_valid_image)

    if len(dataset) == 0:
        raise RuntimeError(
            f"[error] No valid images found in {data_dir}. "
            f"Expected class subfolders with image files: {ALLOWED_EXTS}"
        )

    # Stratified split (best effort)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        # torchvision guarantees .targets for ImageFolder; fallback just in case
        targets = [lbl for _, lbl in dataset.samples]

    train_idx, val_idx = _stratified_indices(targets, test_size=float(test_size), seed=int(seed))
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    if num_workers is None:
        num_workers = _default_num_workers()

    # persistent_workers works only if num_workers > 0
    persistent = bool(num_workers and num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=False,  # cross-platform safe default
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader


def concat_data(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate a whole DataLoader into dense tensors (X, y)."""
    Xs, ys = [], []
    for xb, yb in loader:
        Xs.append(xb)
        ys.append(yb)
    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)