import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

MAIN_NPZ = r"CSV_MINERS\CSV_MINERS\encoded_images.npz"
HARD_NPZ = r"CSV_MINERS\CSV_MINERS\hard_eval_images.npz"

SERVER_IPS = [
    "157.230.14.71", "157.230.14.73", "157.230.6.255", "157.230.10.203",
    "157.230.14.174", "157.230.14.28", "204.48.26.169", "159.89.85.4",
    "147.182.169.120", "161.35.107.208",
]

N_EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SmallCNN(nn.Module):
    """Deliberately small - comparable in spirit to the ~4,500-parameter
    CNN baseline referenced in the original plan, not a large architecture
    that would overfit a 107-positive dataset regardless of regularization."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_fold(X_train, y_train):
    model = SmallCNN().to(DEVICE)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    n = len(X_t)

    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = X_t[idx].to(DEVICE)
            yb = y_t[idx].to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if epoch == 0 or epoch == N_EPOCHS - 1:
            print(f"    epoch {epoch + 1}/{N_EPOCHS}: train_loss={total_loss / n:.4f}")

    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def main():
    print(f"Using device: {DEVICE}")
    main_data = np.load(MAIN_NPZ, allow_pickle=True)
    hard_data = np.load(HARD_NPZ, allow_pickle=True)

    X_main, y_main, g_main = main_data["X_gaf"], main_data["y"], main_data["vps_group"]
    X_hard, y_hard, g_hard = hard_data["X_gaf"], hard_data["y"], hard_data["vps_group"]
    cluster_hard = hard_data["cluster_id"]

    print(f"Main pool: {X_main.shape}, Hard eval: {X_hard.shape}")
    print(f"Model parameter count: {count_params(SmallCNN())}\n")

    oof_main_probs = np.zeros(len(y_main))
    oof_hard_probs = np.zeros(len(y_hard))

    for fold_i, vps in enumerate(SERVER_IPS):
        train_mask = g_main != vps
        test_main_mask = g_main == vps
        test_hard_mask = g_hard == vps

        X_train, y_train = X_main[train_mask], y_main[train_mask]
        print(f"Fold {fold_i} (held out {vps}): "
              f"train n={len(y_train)} (pos={y_train.sum()}), "
              f"main_test n={test_main_mask.sum()}, hard_test n={test_hard_mask.sum()}")

        model = train_one_fold(X_train, y_train)

        oof_main_probs[test_main_mask] = predict(model, X_main[test_main_mask])
        oof_hard_probs[test_hard_mask] = predict(model, X_hard[test_hard_mask])

    print("\n=== Pooled out-of-fold results: MAIN test (easy, vs all sampled normal) ===")
    auc_main = roc_auc_score(y_main, oof_main_probs)
    f1_main = f1_score(y_main, (oof_main_probs >= 0.5).astype(int))
    print(f"AUC={auc_main:.4f}, F1={f1_main:.4f}  (length-only baseline was AUC=0.9954)")

    print("\n=== Pooled out-of-fold results: HARD test (vs length-matched negatives) ===")
    auc_hard = roc_auc_score(y_hard, oof_hard_probs)
    f1_hard = f1_score(y_hard, (oof_hard_probs >= 0.5).astype(int))
    print(f"AUC={auc_hard:.4f}, F1={f1_hard:.4f}  (length-only baseline was AUC=0.5708)")

    print("\n=== Cluster-deduplicated HARD test (one pred per mining signature cluster) ===")
    is_mining = y_hard == 1
    dedup_probs, dedup_labels = [], []
    seen_clusters = set()
    for i in range(len(y_hard)):
        if is_mining[i]:
            c = cluster_hard[i]
            if c in seen_clusters:
                continue
            seen_clusters.add(c)
        dedup_probs.append(oof_hard_probs[i])
        dedup_labels.append(y_hard[i])
    dedup_probs = np.array(dedup_probs)
    dedup_labels = np.array(dedup_labels)
    print(f"Deduplicated set: {len(dedup_labels)} flows "
          f"({dedup_labels.sum()} distinct mining clusters, "
          f"{len(dedup_labels) - dedup_labels.sum()} hard negatives)")
    if dedup_labels.sum() > 0 and (len(dedup_labels) - dedup_labels.sum()) > 0:
        auc_dedup = roc_auc_score(dedup_labels, dedup_probs)
        print(f"AUC={auc_dedup:.4f}  "
              f"(compare to full HARD AUC={auc_hard:.4f} above - a big drop here "
              f"means the model is leaning on repeated templates, not generalizing)")
    else:
        print("Not enough of one class after dedup to compute AUC.")


if __name__ == "__main__":
    main()
