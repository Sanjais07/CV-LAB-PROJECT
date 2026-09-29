import json
import random
import numpy as np
from collections import defaultdict

INPUT_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\modeling_dataset.jsonl"
OUTPUT_NPZ = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\encoded_images.npz"

WINDOW_N = 64          # must match prepare_modeling_dataset.py's WINDOW_N
NEG_PER_GROUP = 2000   # normal flows sampled per VPS group; tune to your RAM/disk
MAX_GAP_SECONDS = 5.0  # clip inter-arrival time before global scaling
RANDOM_SEED = 42

GAF_IMAGE_SIZE = WINDOW_N  # one pixel per packet position, no PAA reduction


def load_and_sample(path, neg_per_group, seed):
    rng = random.Random(seed)
    mining_records = []
    # reservoir sampling per vps_group for normal flows
    reservoirs = defaultdict(list)
    seen_counts = defaultdict(int)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] == "mining" and r["include_as_mining"]:
                mining_records.append(r)
            elif r["label"] == "normal":
                g = r["vps_group"]
                seen_counts[g] += 1
                res = reservoirs[g]
                if len(res) < neg_per_group:
                    res.append(r)
                else:
                    j = rng.randint(0, seen_counts[g] - 1)
                    if j < neg_per_group:
                        res[j] = r

    normal_records = [r for res in reservoirs.values() for r in res]
    print(f"Mining records kept: {len(mining_records)}")
    print(f"Normal records sampled: {len(normal_records)} "
          f"(from groups seen: { {g: c for g, c in seen_counts.items()} })")
    return mining_records + normal_records


def globally_scale(records, window_n, max_gap):
    """Returns three (n_records, window_n) arrays in [-1, 1]: sizes, dirs, gaps."""
    n = len(records)
    sizes_arr = np.empty((n, window_n), dtype=np.float32)
    dirs_arr = np.empty((n, window_n), dtype=np.float32)
    gaps_arr = np.empty((n, window_n), dtype=np.float32)
    mask_arr = np.empty((n, window_n), dtype=np.int8)

    for i, r in enumerate(records):
        # sizes already divided by 1514 in prepare_modeling_dataset.py
        # (approx [0,1], occasionally slightly above for jumbo frames) ->
        # clip then map to [-1, 1]
        sizes = np.clip(np.array(r["sizes"], dtype=np.float32), 0.0, 1.0)
        sizes_arr[i] = 2.0 * sizes - 1.0

        dirs = np.array(r["directions"], dtype=np.float32)  # already 0/1
        dirs_arr[i] = 2.0 * dirs - 1.0

        gaps = np.clip(np.array(r["interarrival_s"], dtype=np.float32), 0.0, max_gap)
        gaps_arr[i] = 2.0 * (gaps / max_gap) - 1.0

        mask_arr[i] = np.array(r["mask"], dtype=np.int8)

    return sizes_arr, dirs_arr, gaps_arr, mask_arr


def encode_all(sizes_arr, dirs_arr, gaps_arr):
    from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

    gaf = GramianAngularField(image_size=GAF_IMAGE_SIZE, sample_range=None,
                               method="summation")
    mtf = MarkovTransitionField(n_bins=8)
    rp = RecurrencePlot()

    channels = {"size": sizes_arr, "direction": dirs_arr, "interarrival": gaps_arr}
    gaf_out, mtf_out, rp_out = [], [], []

    for name, arr in channels.items():
        print(f"Encoding channel '{name}' ({arr.shape}) ...")
        gaf_out.append(gaf.fit_transform(arr))   # (n, W, W)
        mtf_out.append(mtf.fit_transform(arr))   # (n, W, W)
        rp_out.append(rp.fit_transform(arr))     # (n, W, W)

    # stack channels: (n, 3, W, W)
    X_gaf = np.stack(gaf_out, axis=1).astype(np.float32)
    X_mtf = np.stack(mtf_out, axis=1).astype(np.float32)
    X_rp = np.stack(rp_out, axis=1).astype(np.float32)
    return X_gaf, X_mtf, X_rp


def main():
    records = load_and_sample(INPUT_JSONL, NEG_PER_GROUP, RANDOM_SEED)

    labels = np.array([1 if r["label"] == "mining" else 0 for r in records], dtype=np.int8)
    vps_group = np.array([r["vps_group"] for r in records])
    cluster_id = np.array([r["cluster_id"] if r["cluster_id"] is not None else -1
                            for r in records], dtype=np.int32)
    coin = np.array([r["coin"] for r in records])

    sizes_arr, dirs_arr, gaps_arr, mask_arr = globally_scale(records, WINDOW_N, MAX_GAP_SECONDS)

    print(f"\nInput ranges after global scaling (should be within [-1, 1]):")
    print(f"sizes: min={sizes_arr.min():.3f}, max={sizes_arr.max():.3f}")
    print(f"dirs:  min={dirs_arr.min():.3f}, max={dirs_arr.max():.3f}")
    print(f"gaps:  min={gaps_arr.min():.3f}, max={gaps_arr.max():.3f}")

    X_gaf, X_mtf, X_rp = encode_all(sizes_arr, dirs_arr, gaps_arr)

    print(f"\nOutput shapes: GAF={X_gaf.shape}, MTF={X_mtf.shape}, RP={X_rp.shape}")

    np.savez_compressed(
        OUTPUT_NPZ,
        X_gaf=X_gaf, X_mtf=X_mtf, X_rp=X_rp,
        y=labels, vps_group=vps_group, cluster_id=cluster_id, coin=coin,
        mask=mask_arr,
    )
    print(f"\nSaved to {OUTPUT_NPZ}")
    print(f"Positives: {labels.sum()}, Negatives: {len(labels) - labels.sum()}")


if __name__ == "__main__":
    main()
