import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score

INPUT_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\modeling_dataset.jsonl"
HARD_NEG_OUTPUT = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\hard_negatives.jsonl"

HARD_NEG_MIN_PACKETS = 32  # normal flows at/above this are the "hard negative" set
N_SPLITS = 10  # matches the 10 VPS groups (leave-one-VPS-out)


def load_all(path):
    n_packets = []
    labels = []
    groups = []
    all_records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] == "mining" and not r["include_as_mining"]:
                continue  # excluded ETH-short flows, same as everywhere else
            n_packets.append(r["n_packets_real"])
            labels.append(1 if r["label"] == "mining" else 0)
            groups.append(r["vps_group"])
            all_records.append(r)
    return (np.array(n_packets, dtype=np.float64).reshape(-1, 1),
            np.array(labels, dtype=np.int8),
            np.array(groups),
            all_records)


def run_length_only_baseline(X, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    aucs, f1s = [], []
    print(f"=== Length-only baseline: GroupKFold(n_splits={N_SPLITS}) on vps_group ===")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if y_train.sum() == 0 or y_test.sum() == 0:
            print(f"  Fold {fold}: SKIPPED (no positives in train or test - "
                  f"train_pos={y_train.sum()}, test_pos={y_test.sum()})")
            continue

        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)

        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        aucs.append(auc)
        f1s.append(f1)
        held_out_group = np.unique(groups[test_idx])
        print(f"  Fold {fold} (held-out VPS={held_out_group}): "
              f"AUC={auc:.4f}, F1={f1:.4f}, "
              f"train_pos={y_train.sum()}, test_pos={y_test.sum()}")

    print(f"\nMean AUC across valid folds: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
    print(f"Mean F1 across valid folds: {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
    print(
        "\nThis is the bar. Any GAF/MTF/RP + CNN/ViT result at or below "
        "this AUC/F1 has NOT demonstrated learning beyond flow length - "
        "report both numbers side by side in your results table."
    )


def extract_hard_negatives(records, min_packets, output_path):
    hard_negs = [r for r in records
                 if r["label"] == "normal" and r["n_packets_real"] >= min_packets]

    print(f"\n=== Hard-negative set: normal flows with n_packets_real >= {min_packets} ===")
    print(f"Count: {len(hard_negs)}")

    per_group = {}
    for r in hard_negs:
        per_group[r["vps_group"]] = per_group.get(r["vps_group"], 0) + 1
    print(f"Per-VPS breakdown: {per_group}")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in hard_negs:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {output_path}")

    if len(hard_negs) == 0:
        print(
            "\nWARNING: zero hard negatives found at this threshold. Lower "
            "HARD_NEG_MIN_PACKETS and rerun - do not proceed to a "
            "length-matched evaluation with an empty set."
        )


if __name__ == "__main__":
    X, y, groups, records = load_all(INPUT_JSONL)
    print(f"Loaded {len(y)} flows ({y.sum()} mining, {len(y) - y.sum()} normal)\n")

    run_length_only_baseline(X, y, groups)
    extract_hard_negatives(records, HARD_NEG_MIN_PACKETS, HARD_NEG_OUTPUT)
