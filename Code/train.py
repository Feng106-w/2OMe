# train_caps_fusion.py
import os
import copy
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, f1_score,
    roc_auc_score, average_precision_score
)

from data.fused_caps_dataset import FusedCapsDataset, fm_collate
from models.capsnet_8x8 import CapsNet, USE_CUDA
from models.fusion_frontend import FusionToCapsInput


def set_seed(seed=48):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _caps_probs_from_output(caps_out: torch.Tensor) -> torch.Tensor:
    # [B, 2, C, 1] -> [B, 2] -> prob_pos
    lens = torch.sqrt((caps_out ** 2).sum(dim=2)).squeeze(-1)
    lens = torch.nan_to_num(lens, nan=0.0, posinf=1e6, neginf=0.0)
    probs = torch.softmax(lens, dim=1)
    probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
    return probs[:, 1]


def evaluate_caps(model, loader, device):
    model.eval()
    all_pred, all_true, all_prob = [], [], []
    with torch.no_grad():
        for (fm, cgr), y in loader:
            y = y.to(device)
            x = model.front(fm.to(device), cgr.to(device))
            out, rec, masked = model.caps(x)

            pred = torch.argmax(masked, dim=1).float().cpu().numpy()
            prob_pos = _caps_probs_from_output(out).cpu().numpy()

            all_pred.extend(pred.tolist())
            all_true.extend(y.cpu().numpy().tolist())
            all_prob.extend(prob_pos.tolist())

    tn, fp, fn, tp = confusion_matrix(all_true, all_pred, labels=[0, 1]).ravel()
    all_prob = np.clip(
        np.nan_to_num(np.array(all_prob, dtype=float), nan=0.5, posinf=1.0, neginf=0.0),
        0.0, 1.0
    )

    perftab = {
        'ACC': (tp + tn) / (tp + fp + fn + tn + 1e-12),
        'SEN': tp / (tp + fn + 1e-12),
        'PREC': tp / (tp + fp + 1e-12),
        'SPEC': tn / (tn + fp + 1e-12),
        'MCC': matthews_corrcoef(all_true, all_pred) if (len(set(all_true)) == 2 and len(set(all_pred)) == 2) else 0.0,
        'F1': f1_score(all_true, all_pred) if len(set(all_true)) == 2 else 0.0,
        'AUC': roc_auc_score(all_true, all_prob) if len(set(all_true)) == 2 else 0.5,
        'AUPR': average_precision_score(all_true, all_prob) if len(set(all_true)) == 2 else 0.5,
        'CM': np.array([[tn, fp], [fn, tp]], dtype=int),
        'y_true': np.array(all_true, dtype=int),
        'y_prob': np.array(all_prob, dtype=float)
    }
    return perftab


class FusedCapsModel(torch.nn.Module):
    def __init__(self, primary_caps_num=8):
        super().__init__()
        self.front = FusionToCapsInput()
        self.caps = CapsNet(Primary_capsule_num=primary_caps_num)

    def forward(self, fm, cgr):
        x = self.front(fm, cgr)
        out, rec, masked = self.caps(x)
        return out, rec, masked, x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fasta", type=str, default="data/Benchmark_Set.fasta")
    parser.add_argument("--test_fasta",  type=str, default="data/Independent_Test_Set.fasta")
    parser.add_argument("--train_cgr",   type=str, default="data/Benchmark_Set8.txt")
    parser.add_argument("--test_cgr",    type=str, default="data/Independent_Test_Set8.txt")
    parser.add_argument("--pretrained_dir", type=str, default="models_folder")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on Validation ACC")
    parser.add_argument("--caps_nums", type=str, default="4,8,16,32")
    parser.add_argument("--save_dir", type=str, default="checkpoints_caps")
    parser.add_argument("--pred_csv", type=str, default="results/test_pred.csv",
                        help="The CSV path for saving the test set probabilities and labels")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.pred_csv), exist_ok=True)

    print("==> Building datasets (RNA-FM batched online + in-memory cache + CGR 8x8)")

    train_set = FusedCapsDataset(args.train_fasta, args.train_cgr, args.pretrained_dir, device)
    test_set  = FusedCapsDataset(args.test_fasta,  args.test_cgr,  args.pretrained_dir, device)

    from data.fused_caps_dataset import FM_CACHE
    X_idx, y_all = train_set.get_all_features_labels()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # The global optimum is also selected based on ACC
    best_global = {"acc": 0.0, "path": "", "caps": 0, "fold": 0}
    caps_list = [int(x) for x in args.caps_nums.split(",") if x.strip()]

    for primary_caps_num in caps_list:
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_idx, y_all), start=1):
            print(f"\n=== Training: Caps={primary_caps_num}, Fold={fold} ===")

            tr_loader = DataLoader(
                Subset(train_set, tr_idx),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=fm_collate
            )
            va_loader = DataLoader(
                Subset(train_set, va_idx),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=fm_collate
            )

            model = FusedCapsModel(primary_caps_num=primary_caps_num)
            if device == "cuda":
                model = model.cuda()

            opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

            best_acc = -1.0
            best_state = None
            best_epoch = 0
            epochs_no_improve = 0

            for epoch in range(1, args.epochs + 1):
                model.train()
                total_loss = 0.0

                for (fm, cgr), y in tr_loader:
                    y_long = y.long()
                    y_oh = torch.eye(2).index_select(0, y_long)

                    if device == "cuda":
                        fm, cgr, y_oh = fm.cuda(), cgr.cuda(), y_oh.cuda()

                    opt.zero_grad()
                    out, rec, masked, ximg = model(fm, cgr)
                    loss = model.caps.loss(ximg, out, y_oh, rec)
                    loss.backward()
                    opt.step()

                    total_loss += loss.item() * y.size(0)

                perf = evaluate_caps(model, va_loader, device)
                val_acc = perf["ACC"]

                print(
                    f"[Epoch {epoch:03d}] "
                    f"TrainLoss={total_loss / max(len(tr_loader.dataset), 1):.4f} | "
                    f"Val ACC={perf['ACC']:.4f} SEN={perf['SEN']:.4f} SPEC={perf['SPEC']:.4f} "
                    f"MCC={perf['MCC']:.4f} F1={perf['F1']:.4f} AUC={perf['AUC']:.4f} AUPR={perf['AUPR']:.4f} "
                    f"| bestACC={best_acc:.4f} no_improve={epochs_no_improve}/{args.patience} "
                    f"| FM_CACHE={len(FM_CACHE)}"
                )

                # Early stopping based on Validation ACC
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improve = 0
                    print(f"  -> New best Val ACC: {best_acc:.4f} at epoch {best_epoch}")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= args.patience:
                    print(
                        f"  -> Early stopping triggered at epoch {epoch}. "
                        f"Best Val ACC={best_acc:.4f} (epoch {best_epoch})"
                    )
                    break

            tag = f"caps{primary_caps_num}_fold{fold}_acc{best_acc:.4f}.pth"
            save_path = os.path.join(args.save_dir, tag)
            torch.save(best_state, save_path)
            print(f"Saved best model: {save_path}")

            if best_acc > best_global["acc"]:
                best_global = {
                    "acc": best_acc,
                    "path": save_path,
                    "caps": primary_caps_num,
                    "fold": fold
                }

    print(f"\n>>> Best overall model: {best_global['path']} (ACC={best_global['acc']:.4f})")

    # ====== Test set evaluation + Probability preservation ======
    print("\n==> Evaluating on test set ...")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=fm_collate
    )

    best_model = FusedCapsModel(primary_caps_num=best_global["caps"])
    best_model.load_state_dict(torch.load(best_global["path"], map_location=device))
    if device == "cuda":
        best_model = best_model.cuda()

    best_model.eval()
    all_pred, all_true, all_prob = [], [], []
    with torch.no_grad():
        for (fm, cgr), y in test_loader:
            if device == "cuda":
                fm, cgr = fm.cuda(), cgr.cuda()

            out, rec, masked, ximg = best_model(fm, cgr)
            pred = torch.argmax(masked, dim=1).cpu().numpy()
            prob_pos = _caps_probs_from_output(out).cpu().numpy()

            all_pred.extend(pred.tolist())
            all_true.extend(y.numpy().tolist())
            all_prob.extend(prob_pos.tolist())

    tn, fp, fn, tp = confusion_matrix(all_true, all_pred, labels=[0, 1]).ravel()
    ACC  = (tp + tn) / (tp + fp + fn + tn + 1e-12)
    REC  = tp / (tp + fn + 1e-12)
    SPEC = tn / (tn + fp + 1e-12)
    PREC = tp / (tp + fp + 1e-12)
    MCC  = matthews_corrcoef(all_true, all_pred) if (len(set(all_true)) == 2 and len(set(all_pred)) == 2) else 0.0
    F1   = f1_score(all_true, all_pred) if len(set(all_true)) == 2 else 0.0
    AUC  = roc_auc_score(all_true, all_prob) if len(set(all_true)) == 2 else 0.5
    AUPR = average_precision_score(all_true, all_prob) if len(set(all_true)) == 2 else 0.5

    print(
        f"ACC={ACC:.4f}  F1={F1:.4f}  Precision={PREC:.4f}  Recall={REC:.4f}  "
        f"AUC={AUC:.4f}  AUPR={AUPR:.4f}  MCC={MCC:.4f}"
    )
    print(f"Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

    import pandas as pd
    df = pd.DataFrame({
        "y_true": np.array(all_true, dtype=int),
        "prob_pos": np.array(all_prob, dtype=float)
    })
    df.to_csv(args.pred_csv, index=False)
    print(f"Saved test probabilities to: {args.pred_csv}")
    print("Done.")


if __name__ == "__main__":
    main()