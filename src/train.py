import numpy as np
import torch
import torch.nn.functional as F

from models import ConvNet
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils import eval_metrics, best_device
from data import concat_data



def train_cnn(train_loader, val_loader, args):
    device = best_device()
    print(f"[device] {device}")
    model = ConvNet(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        epochs=args.epochs,
        lr=args.lr,
        class_weights=[args.w0, args.w1],
        weight_decay=args.weight_decay,
    ).to(device)

    optim = model.configure_optimizer(name=args.opt, lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for ep in range(1, args.epochs + 1):
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # [B]
            w = torch.where(yb == 0, torch.tensor(args.w0, device=device), torch.tensor(args.w1, device=device))
            loss = F.binary_cross_entropy_with_logits(logits, yb.float(), weight=w)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # detach to avoid "requires_grad=True to scalar" warning
            running += loss.detach().item() * xb.size(0)
            n += xb.size(0)
        print(f"[cnn][epoch {ep}] loss={running / max(1, n):.4f}")

    # ----- evaluation -----
    X_val, y_val = concat_data(val_loader)
    with torch.no_grad():
        probs_t = model.predict_proba(X_val.to(device)).detach().cpu()  # [N]
    probs = probs_t.numpy()
    preds = (probs > 0.5).astype(np.int64)
    metrics = eval_metrics(y_val.numpy(), preds, probs)
    return model, metrics, preds, probs, y_val.numpy()


def train_rf(train_loader, val_loader, args):
    X_tr, y_tr = concat_data(train_loader)
    X_va, y_va = concat_data(val_loader)
    X_tr = X_tr.view(len(y_tr), -1).numpy()
    X_va = X_va.view(len(y_va), -1).numpy()

    rf = RandomForestClassifier(random_state=args.seed)
    rf.fit(X_tr, y_tr.numpy())

    probs = rf.predict_proba(X_va)[:, 1]
    preds = (probs > 0.5).astype(np.int64)
    metrics = eval_metrics(y_va.numpy(), preds, probs)
    return rf, metrics, preds, probs, y_va.numpy()


def train_xgb(train_loader, val_loader, args):
    X_tr, y_tr = concat_data(train_loader)
    X_va, y_va = concat_data(val_loader)
    X_tr = X_tr.view(len(y_tr), -1).numpy()
    X_va = X_va.view(len(y_va), -1).numpy()

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=4,
        reg_lambda=1.0,
    )
    xgb.fit(X_tr, y_tr.numpy())

    probs = xgb.predict_proba(X_va)[:, 1]
    preds = (probs > 0.5).astype(np.int64)
    metrics = eval_metrics(y_va.numpy(), preds, probs)
    return xgb, metrics, preds, probs, y_va.numpy()