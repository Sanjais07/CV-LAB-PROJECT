import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score

MODELING_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\modeling_dataset.jsonl"
HARD_NEG_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\hard_negatives.jsonl"

# Mining percentiles from the previous run - hardcoded here so this script
# tests against the ACTUAL observed mining distribution, not a guess.
MINING_PERCENTILE_THRESHOLDS = {"p10": 257, "p25": 413, "p50": 1151}

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


def run_baseline(records, label_name):
    X = np.array([r["n_packets_real"] for r in records], dtype=np.float64).reshape(-1, 1)
    y = np.array([1 if r["label"] == "mining" else 0 for r in records], dtype=np.int8)
    groups = np.array([r["vps_group"] for r in records])

    print(f"\n=== Length-only baseline: {label_name} ===")
    print(f"n={len(records)}, mining={y.sum()}, normal={len(y) - y.sum()}")

    n_groups_present = len(np.unique(groups))
    if n_groups_present < 2:
        print("  Fewer than 2 distinct VPS groups present - cannot run GroupKFold.")
        return

    gkf = GroupKFold(n_splits=min(N_SPLITS, n_groups_present))
    aucs, f1s = [], []
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
        print("  No valid folds.")


def main():
    mining = load_mining(MODELING_JSONL)
    hard_neg = load_hard_negatives(HARD_NEG_JSONL)
    print(f"Loaded {len(mining)} mining flows, {len(hard_neg)} hard-negative flows")

    print("\n=== Step 1: how many hard negatives are genuinely competitive in length? ===")
    for name, threshold in MINING_PERCENTILE_THRESHOLDS.items():
        count = sum(1 for r in hard_neg if r["n_packets_real"] >= threshold)
        print(f"Hard negatives with n_packets_real >= {threshold} "
              f"(mining's {name}): {count} out of {len(hard_neg)}")

    genuinely_long = [r for r in hard_neg
                      if r["n_packets_real"] >= MINING_PERCENTILE_THRESHOLDS["p10"]]
    print(f"\nUsing threshold >= {MINING_PERCENTILE_THRESHOLDS['p10']} (mining's p10) "
          f"as the genuinely-competitive hard-negative set: {len(genuinely_long)} flows")

    if len(genuinely_long) < 10:
        print(
            "\nFewer than 10 genuinely-competitive negatives exist. This IS the "
            "finding: essentially no normal traffic in this dataset reaches the "
            "length scale of a typical mining session. Full-flow length is a "
            "near-perfect, legitimate discriminator here - the open research "
            "question is early/partial-flow detection (classifying from the "
            "first N packets), not full-flow classification. Skipping the "
            "baseline rerun (too few samples for a meaningful result)."
        )
    else:
        run_baseline(mining + genuinely_long,
                     f"mining vs genuinely-long negatives (n_real >= {MINING_PERCENTILE_THRESHOLDS['p10']})")

    print("\n=== Step 2: details of the single longest hard-negative flow ===")
    longest = max(hard_neg, key=lambda r: r["n_packets_real"])
    print(f"vps_group: {longest['vps_group']}")
    print(f"n_packets_real: {longest['n_packets_real']}")
    print(f"label: {longest['label']}, flag: {longest['flag']}")
    print(
        "\nTo find this flow's actual IP/port, search flow_sequences.jsonl for "
        f"a record with label=='normal', a matching vps_group of "
        f"'{longest['vps_group']}', and n_packets == {longest['n_packets_real']} "
        "(this combination is very likely unique given how extreme the value is)."
    )


if __name__ == "__main__":
    main()
