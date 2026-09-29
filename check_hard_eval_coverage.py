import numpy as np

MAIN_NPZ = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\encoded_images.npz"
HARD_NPZ = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\hard_eval_images.npz"

SERVER_IPS = [
    "157.230.14.71", "157.230.14.73", "157.230.6.255", "157.230.10.203",
    "157.230.14.174", "157.230.14.28", "204.48.26.169", "159.89.85.4",
    "147.182.169.120", "161.35.107.208",
]


def main():
    main_data = np.load(MAIN_NPZ, allow_pickle=True)
    hard_data = np.load(HARD_NPZ, allow_pickle=True)

    main_y = main_data["y"]
    main_groups = main_data["vps_group"]
    hard_y = hard_data["y"]
    hard_groups = hard_data["vps_group"]

    print(f"Main pool (encoded_images.npz): {len(main_y)} flows, "
          f"{main_y.sum()} mining, {len(main_y) - main_y.sum()} normal")
    print(f"Hard eval (hard_eval_images.npz): {len(hard_y)} flows, "
          f"{hard_y.sum()} mining, {len(hard_y) - hard_y.sum()} normal\n")

    print(f"{'VPS':<18} {'HardEval_mining':>16} {'HardEval_normal':>16} "
          f"{'TrainPool_mining(other9)':>26} {'TrainPool_normal(other9)':>26} {'FOLD_OK':>8}")

    usable_folds = 0
    for vps in SERVER_IPS:
        he_mask = hard_groups == vps
        he_mining = int(hard_y[he_mask].sum())
        he_normal = int((hard_y[he_mask] == 0).sum())

        train_mask = main_groups != vps  # other 9 VPS
        train_mining = int(main_y[train_mask].sum())
        train_normal = int((main_y[train_mask] == 0).sum())

        fold_ok = he_mining > 0 and he_normal > 0 and train_mining > 0
        if fold_ok:
            usable_folds += 1

        print(f"{vps:<18} {he_mining:>16} {he_normal:>16} "
              f"{train_mining:>26} {train_normal:>26} {'YES' if fold_ok else 'NO':>8}")

    print(f"\nUsable leave-one-VPS-out folds for the hard-eval comparison: "
          f"{usable_folds} out of {len(SERVER_IPS)}")
    if usable_folds < len(SERVER_IPS):
        print(
            "Some folds are NOT usable (missing mining or hard-negative "
            "examples for that held-out VPS). Report results only over the "
            "usable folds, and say explicitly how many VPS were excluded "
            "and why - do not average in a fold with an undefined AUC."
        )


if __name__ == "__main__":
    main()
