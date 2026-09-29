import json
import random
import warnings
import numpy as np
from collections import defaultdict

INPUT_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\modeling_dataset.jsonl"
OUTPUT_NPZ = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\encoded_mtf_fixed.npz"

WINDOW_N = 64
NEG_PER_GROUP = 2000
MAX_GAP_SECONDS = 5.0
RANDOM_SEED = 42  # MUST match encode_images.py to reproduce the same sample

N_BINS_CANDIDATES = [8, 4, 3, 2]


def load_and_sample(path, neg_per_group, seed):
    """Identical logic to encode_images.py's load_and_sample."""
    rng = random.Random(seed)
    mining_records = []
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
    return mining_records + normal_records


def globally_scale(records, window_n, max_gap):
    n = len(records)
    sizes_arr = np.empty((n, window_n), dtype=np.float32)
    dirs_arr = np.empty((n, window_n), dtype=np.float32)
    gaps_arr = np.empty((n, window_n), dtype=np.float32)
    mask_arr = np.empty((n, window_n), dtype=np.int8)

    for i, r in enumerate(records):
        sizes = np.clip(np.array(r["sizes"], dtype=np.float32), 0.0, 1.0)
        sizes_arr[i] = 2.0 * sizes - 1.0
        dirs = np.array(r["directions"], dtype=np.float32)
        dirs_arr[i] = 2.0 * dirs - 1.0
        gaps = np.clip(np.array(r["interarrival_s"], dtype=np.float32), 0.0, max_gap)
        gaps_arr[i] = 2.0 * (gaps / max_gap) - 1.0
        mask_arr[i] = np.array(r["mask"], dtype=np.int8)

    return sizes_arr, dirs_arr, gaps_arr, mask_arr


def report_real_packet_distribution(mask_arr):
    n_real = mask_arr.sum(axis=1)  # real (non-padding) packets per flow
    print("=== Distribution of REAL (non-padding) packet counts per sampled flow ===")
    for threshold in [2, 3, 4, 8, 16, 32, 64]:
        pct = 100 * np.mean(n_real <= threshold)
        print(f"  n_real <= {threshold}: {pct:.2f}% of flows")
    print(f"  min={n_real.min()}, median={int(np.median(n_real))}, max={n_real.max()}\n")


def count_warnings_for_nbins(arr, channel_name, n_bins):
    from pyts.image import MarkovTransitionField
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mtf = MarkovTransitionField(n_bins=n_bins)
        out = mtf.fit_transform(arr)
        n_warnings = sum(1 for w in caught if "quantiles are equal" in str(w.message))
    print(f"  channel={channel_name:<13} n_bins={n_bins}: {n_warnings} warning(s) raised")
    return out, n_warnings


def main():
    records = load_and_sample(INPUT_JSONL, NEG_PER_GROUP, RANDOM_SEED)
    print(f"Reproduced sample: {len(records)} flows "
          f"({sum(1 for r in records if r['label']=='mining')} mining)")

    sizes_arr, dirs_arr, gaps_arr, mask_arr = globally_scale(records, WINDOW_N, MAX_GAP_SECONDS)
    report_real_packet_distribution(mask_arr)

    channels = {"size": sizes_arr, "direction": dirs_arr, "interarrival": gaps_arr}

    print("=== Warning counts by n_bins (lower is better; 0 = no collapse at all) ===")
    best_n_bins = {}
    best_output = {}
    for name, arr in channels.items():
        for n_bins in N_BINS_CANDIDATES:
            out, n_warn = count_warnings_for_nbins(arr, name, n_bins)
            if name not in best_n_bins or n_warn == 0:
                # keep the LARGEST n_bins that still gives 0 warnings, so we
                # don't over-simplify the encoding more than necessary
                if n_warn == 0:
                    best_n_bins[name] = n_bins
                    best_output[name] = out
        if name not in best_n_bins:
            # nothing tested reached 0 warnings; fall back to the smallest
            # candidate tested and say so explicitly
            best_n_bins[name] = N_BINS_CANDIDATES[-1]
            print(f"  WARNING: no n_bins in {N_BINS_CANDIDATES} fully eliminated "
                  f"the warning for channel '{name}'. Using smallest tested "
                  f"({N_BINS_CANDIDATES[-1]}) anyway - inspect this channel further.")

    print(f"\nChosen n_bins per channel: {best_n_bins}")

    X_mtf = np.stack([best_output[name] for name in ("size", "direction", "interarrival")],
                      axis=1).astype(np.float32)
    print(f"Final MTF output shape: {X_mtf.shape}")

    labels = np.array([1 if r["label"] == "mining" else 0 for r in records], dtype=np.int8)
    vps_group = np.array([r["vps_group"] for r in records])

    np.savez_compressed(OUTPUT_NPZ, X_mtf=X_mtf, y=labels, vps_group=vps_group,
                         n_bins_used=np.array([best_n_bins[c] for c in
                                                ("size", "direction", "interarrival")]))
    print(f"\nSaved corrected MTF to {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
