import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score

MODELING_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\modeling_dataset.jsonl"
HARD_NEG_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\hard_negatives.jsonl"

N_SPLITS = 10


def load_mining(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] == "mining" and r["include_as_mining"]:
                records.append(r)
    return records


def load_hard_negatives(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def percentile_report(name, values):
    values = np.array(values)
    pct = [0, 10, 25, 50, 75, 90, 100]
    print(f"{name} (n={len(values)}): " +
          ", ".join(f"p{p}={int(np.percentile(values, p))}" for p in pct))


def run_baseline(records, label_name="mining vs hard negatives"):
    X = np.array([r["n_packets_real"] for r in records], dtype=np.float64).reshape(-1, 1)
    y = np.array([1 if r["label"] == "mining" else 0 for r in records], dtype=np.int8)
    groups = np.array([r["vps_group"] for r in records])

    gkf = GroupKFold(n_splits=N_SPLITS)
    aucs, f1s = [], []
    print(f"\n=== Length-only baseline: {label_name} ===")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        y_train, y_test = y[train_idx], y[test_idx]
        if y_train.sum() == 0 or y_test.sum() == 0:
            print(f"  Fold {fold}: SKIPPED (train_pos={y_train.sum()}, test_pos={y_test.sum()})")
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X[train_idx], y_train)
        probs = clf.predict_proba(X[test_idx])[:, 1]
        preds = clf.predict(X[test_idx])
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        aucs.append(auc)
        f1s.append(f1)
        print(f"  Fold {fold}: AUC={auc:.4f}, F1={f1:.4f}, "
              f"train_pos={y_train.sum()}, test_pos={y_test.sum()}")

    if aucs:
        print(f"  Mean AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
        print(f"  Mean F1:  {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
    else:
        print("  No valid folds (every fold had 0 positives somewhere).")
    return aucs, f1s


def main():
    mining = load_mining(MODELING_JSONL)
    hard_neg = load_hard_negatives(HARD_NEG_JSONL)

    print(f"Loaded {len(mining)} mining flows, {len(hard_neg)} hard-negative flows\n")

    print("=== Step 1: packet-count percentile comparison ===")
    percentile_report("Mining", [r["n_packets_real"] for r in mining])
    percentile_report("Hard negatives", [r["n_packets_real"] for r in hard_neg])

    print()
    run_baseline(mining + hard_neg, "mining vs ALL hard negatives (n_real >= 32)")

    # Step 3: restrict both classes to the overlapping packet-count range
    mining_counts = [r["n_packets_real"] for r in mining]
    hardneg_counts = [r["n_packets_real"] for r in hard_neg]
    overlap_lo = max(min(mining_counts), min(hardneg_counts))
    overlap_hi = min(max(mining_counts), max(hardneg_counts))
    print(f"\nOverlap range (both classes present): [{overlap_lo}, {overlap_hi}]")

    if overlap_lo >= overlap_hi:
        print("No overlap in packet-count range - the classes are fully "
              "separable by length alone, by construction. Skipping Step 3.")
        return

    mining_overlap = [r for r in mining if overlap_lo <= r["n_packets_real"] <= overlap_hi]
    hardneg_overlap = [r for r in hard_neg if overlap_lo <= r["n_packets_real"] <= overlap_hi]
    print(f"Mining flows in overlap range: {len(mining_overlap)}")
    print(f"Hard-negative flows in overlap range: {len(hardneg_overlap)}")

    if len(mining_overlap) < 5 or len(hardneg_overlap) < 5:
        print("Too few flows in the overlap range for a meaningful test "
              "(need at least ~5 per class). Reporting counts only.")
        return

    run_baseline(mining_overlap + hardneg_overlap,
                 "mining vs hard negatives, RESTRICTED TO OVERLAPPING packet-count range")


if __name__ == "__main__":
    main()
