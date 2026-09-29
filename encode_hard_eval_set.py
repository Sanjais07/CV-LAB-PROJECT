import json
import numpy as np

MODELING_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\modeling_dataset.jsonl"
HARD_NEG_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\hard_negatives.jsonl"
FLOW_SEQUENCES_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_sequences.jsonl"
OUTPUT_NPZ = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\hard_eval_images.npz"

WINDOW_N = 64
MAX_GAP_SECONDS = 5.0
HARD_NEG_MIN_PACKETS = 257  # mining's p10, per the prior analysis
MTF_N_BINS = 2  # per fix_mtf_encoding.py's finding


def find_outlier_flow(path, target_vps_group, target_n_packets):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] != "normal" or r["n_packets"] != target_n_packets:
                continue
            server_is_a = r["ip_a"] == target_vps_group
            server_is_b = r["ip_b"] == target_vps_group
            if not (server_is_a or server_is_b):
                continue
            print("=== Outlier flow found ===")
            print(f"ip_a:port_a = {r['ip_a']}:{r['port_a']}")
            print(f"ip_b:port_b = {r['ip_b']}:{r['port_b']}")
            print(f"n_packets = {r['n_packets']}")
            remote_ip, remote_port = (
                (r["ip_b"], r["port_b"]) if server_is_a else (r["ip_a"], r["port_a"])
            )
            print(f"Remote endpoint: {remote_ip}:{remote_port} "
                  f"(look this port up manually - common long-lived legitimate "
                  f"services include persistent monitoring/keep-alive connections, "
                  f"but do not assume without checking)")
            return r
    print("Outlier flow not found with exact match - check target values.")
    return None


def load_mining(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] == "mining" and r["include_as_mining"]:
                records.append(r)
    return records


def load_hard_negatives_filtered(path, min_packets):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["n_packets_real"] >= min_packets:
                records.append(r)
    return records


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


def encode_all(sizes_arr, dirs_arr, gaps_arr):
    from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

    gaf = GramianAngularField(image_size=sizes_arr.shape[1], sample_range=None,
                               method="summation")
    mtf = MarkovTransitionField(n_bins=MTF_N_BINS)
    rp = RecurrencePlot()

    channels = {"size": sizes_arr, "direction": dirs_arr, "interarrival": gaps_arr}
    gaf_out, mtf_out, rp_out = [], [], []
    for name, arr in channels.items():
        print(f"Encoding channel '{name}' ({arr.shape}) ...")
        gaf_out.append(gaf.fit_transform(arr))
        mtf_out.append(mtf.fit_transform(arr))
        rp_out.append(rp.fit_transform(arr))

    X_gaf = np.stack(gaf_out, axis=1).astype(np.float32)
    X_mtf = np.stack(mtf_out, axis=1).astype(np.float32)
    X_rp = np.stack(rp_out, axis=1).astype(np.float32)
    return X_gaf, X_mtf, X_rp


def main():
    find_outlier_flow(FLOW_SEQUENCES_JSONL, "157.230.10.203", 268900)

    print("\n=== Building hard-evaluation image set ===")
    mining = load_mining(MODELING_JSONL)
    hard_neg = load_hard_negatives_filtered(HARD_NEG_JSONL, HARD_NEG_MIN_PACKETS)
    records = mining + hard_neg
    print(f"Mining: {len(mining)}, hard negatives (n_real >= {HARD_NEG_MIN_PACKETS}): {len(hard_neg)}")

    labels = np.array([1 if r["label"] == "mining" else 0 for r in records], dtype=np.int8)
    vps_group = np.array([r["vps_group"] for r in records])
    cluster_id = np.array([r["cluster_id"] if r["cluster_id"] is not None else -1
                            for r in records], dtype=np.int32)

    sizes_arr, dirs_arr, gaps_arr, mask_arr = globally_scale(records, WINDOW_N, MAX_GAP_SECONDS)
    X_gaf, X_mtf, X_rp = encode_all(sizes_arr, dirs_arr, gaps_arr)

    print(f"\nOutput shapes: GAF={X_gaf.shape}, MTF={X_mtf.shape}, RP={X_rp.shape}")

    np.savez_compressed(
        OUTPUT_NPZ,
        X_gaf=X_gaf, X_mtf=X_mtf, X_rp=X_rp,
        y=labels, vps_group=vps_group, cluster_id=cluster_id, mask=mask_arr,
    )
    print(f"Saved to {OUTPUT_NPZ}")
    print(f"Positives: {labels.sum()}, Negatives: {len(labels) - labels.sum()}")
    print(
        "\nThis is your headline secondary result: whatever AUC/F1 your CNN/ViT "
        "gets on this file, compare it directly against the 0.5708 AUC length-only "
        "baseline on this same 185-flow comparison."
    )


if __name__ == "__main__":
    main()
