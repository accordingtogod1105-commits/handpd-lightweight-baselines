import os, argparse
import numpy as np

from data import download_handpd, load_dataloader
from utils import save_confusion_matrix, save_sample_preview, set_seed, ensure_dir, dump_json
from train import train_cnn, train_rf, train_xgb


def main():
    p = argparse.ArgumentParser(description="HandPD Lightweight Baselines")
    p.add_argument("--dataset", choices=["Meander", "Spiral"], default="Meander")
    p.add_argument("--data_root", default="./data/Handpd")
    p.add_argument("--download", action="store_true")
    p.add_argument("--models", nargs="+", default=["cnn", "rf", "xgb"], choices=["cnn", "rf", "xgb"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--opt", type=str, default="adam", choices=["adam","adamw","sgd"])
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--w0", type=float, default=0.5)
    p.add_argument("--w1", type=float, default=0.5)
    p.add_argument("--img_size", type=int, default=28)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs/run")
    args = p.parse_args()

    set_seed(args.seed)

    if args.download:
        download_handpd(args.data_root)

    data_dir = os.path.join(args.data_root, f"{args.dataset}_HandPD")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"[error] Data not found at: {data_dir}. Run with --download or check path.")

    ensure_dir(args.output_dir)
    save_sample_preview(data_dir, os.path.join(args.output_dir, "sample_preview.png"), img_size=args.img_size)

    train_loader, val_loader = load_dataloader(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    dump_json(vars(args), os.path.join(args.output_dir, "run_config.json"))

    if "cnn" in args.models:
        model, metrics, preds, probs, y_true = train_cnn(train_loader, val_loader, args)
        dump_json(metrics, os.path.join(args.output_dir, "metrics_cnn.json"))
        np.savetxt(os.path.join(args.output_dir, "preds_cnn.csv"),
                   np.column_stack([y_true, probs, preds]),
                   delimiter=",", header="y_true,y_prob,y_pred", comments="")
        save_confusion_matrix(y_true, preds, os.path.join(args.output_dir, "cm_cnn.png"), title="CNN Confusion Matrix")
        print("[cnn]", metrics)

    if "rf" in args.models:
        model, metrics, preds, probs, y_true = train_rf(train_loader, val_loader, args)
        dump_json(metrics, os.path.join(args.output_dir, "metrics_rf.json"))
        np.savetxt(os.path.join(args.output_dir, "preds_rf.csv"),
                   np.column_stack([y_true, probs, preds]),
                   delimiter=",", header="y_true,y_prob,y_pred", comments="")
        save_confusion_matrix(y_true, preds, os.path.join(args.output_dir, "cm_rf.png"), title="RF Confusion Matrix")
        print("[rf ]", metrics)

    if "xgb" in args.models:
        model, metrics, preds, probs, y_true = train_xgb(train_loader, val_loader, args)
        dump_json(metrics, os.path.join(args.output_dir, "metrics_xgb.json"))
        np.savetxt(os.path.join(args.output_dir, "preds_xgb.csv"),
                   np.column_stack([y_true, probs, preds]),
                   delimiter=",", header="y_true,y_prob,y_pred", comments="")
        save_confusion_matrix(y_true, preds, os.path.join(args.output_dir, "cm_xgb.png"), title="XGB Confusion Matrix")
        print("[xgb]", metrics)

if __name__ == "__main__":
    main()