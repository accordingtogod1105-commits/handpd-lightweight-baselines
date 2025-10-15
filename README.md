# Feasibility of Image-Only Parkinson’s Screening from Scanned Spirals and Meanders: A Comparative Study on HandPD.

Minimal, reproducible baselines for image-only Parkinson’s disease screening from hand-drawn spiral and meander images (HandPD dataset).
Implements lightweight CNN and tree-based ensembles under small-data constraints — emphasizing feasibility, not hyperparameter optimization.

Single-folder, notebook-free structure for easy reproducibility:
run.py, data.py, models.py, train.py, utils.py.

All figures and metrics are automatically saved to disk for analysis.

## Structure
```text
.
├─ run.py          # Main CLI entry point
├─ data.py         # Dataset download & loader
├─ models.py       # Lightweight 1-channel ConvNet + wrappers for RF/XGB
├─ train.py        # Train & evaluate across CNN / RF / XGB
└─ utils.py        # Reproducibility, device setup, save CM/preview/JSON
```

## Setup
```bash
# Conda (recommended)
conda create -n handpd python=3.10 -y
conda activate handpd

# Install dependencies
pip install torch torchvision numpy matplotlib tqdm scikit-learn xgboost Pillow
```

## Quickstart
``` bash
# 1) Download HandPD dataset into ./data/HandPD
python src/run.py --download

# 2) Run baselines on Meander
python src/run.py --dataset Meander --models cnn rf xgb --epochs 5 --img_size 28 --output_dir outputs/meander

# 3) Run on Spiral
python src/run.py --dataset Spiral --models cnn xgb --epochs 5 --img_size 28 --output_dir outputs/spiral
```
