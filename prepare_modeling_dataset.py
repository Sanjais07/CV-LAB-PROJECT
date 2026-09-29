import json
from collections import defaultdict

INPUT_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_sequences.jsonl"
OUTPUT_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\modeling_dataset.jsonl"

SERVER_IPS = {
    "157.230.14.71", "157.230.14.73", "157.230.6.255", "157.230.10.203",
    "157.230.14.174", "157.230.14.28", "204.48.26.169", "159.89.85.4",
    "147.182.169.120", "161.35.107.208",
}

WINDOW_N = 64        # fixed sequence length for GAF/MTF encoding
SIG_LEN = 16         # packets used for near-duplicate signature
SHORT_MAX = 31       # <=31 packets counts as "short"
MAX_FRAME_LEN = 1514.0  # standard Ethernet MTU, for global size scaling


def pad_or_truncate(seq, n, fill=0.0):
    seq = seq[:n]
    mask = [1] * len(seq)
    if len(seq) < n:
        pad_len = n - len(seq)
        seq = seq + [fill] * pad_len
        mask = mask + [0] * pad_len
    return seq, mask


def main():
    # First pass: read everything, compute mining-flow signatures for
    # clustering (needs to see all mining flows before assigning cluster
    # ids consistently).
    records = []
    mining_signatures = defaultdict(list)  # signature -> list of record indices

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append(r)

    print(f"Read {len(records)} flows from {INPUT_JSONL}")

    for idx, r in enumerate(records):
        if r["label"] != "mining":
            continue
        server_is_a = r["ip_a"] in SERVER_IPS
        rel_dirs = [(d == 1) == server_is_a for d in r["directions"][:SIG_LEN]]
        sig = tuple(zip(r["packet_sizes"][:SIG_LEN], rel_dirs))
        mining_signatures[sig].append(idx)

    # Assign a stable cluster id per distinct signature, sorted by size
    # descending so cluster 0 is the largest (most duplicated) group.
    sorted_sigs = sorted(mining_signatures.items(), key=lambda kv: -len(kv[1]))
    idx_to_cluster = {}
    for cluster_id, (sig, idx_list) in enumerate(sorted_sigs):
        for idx in idx_list:
            idx_to_cluster[idx] = cluster_id

    print(f"Mining flows grouped into {len(sorted_sigs)} signature clusters")

    n_written = 0
    n_eth_short_flagged = 0
    group_label_counts = defaultdict(lambda: defaultdict(int))

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for idx, r in enumerate(records):
            server_is_a = r["ip_a"] in SERVER_IPS
            vps_group = r["ip_a"] if server_is_a else r["ip_b"]

            rel_directions = [
                1 if (d == 1) == server_is_a else 0 for d in r["directions"]
            ]
            scaled_sizes = [s / MAX_FRAME_LEN for s in r["packet_sizes"]]

            sizes_p, mask = pad_or_truncate(scaled_sizes, WINDOW_N)
            dirs_p, _ = pad_or_truncate(rel_directions, WINDOW_N, fill=0)
            gaps_p, _ = pad_or_truncate(r["interarrival_s"], WINDOW_N)

            include_as_mining = True
            flag = None
            cluster_id = None

            if r["label"] == "mining":
                cluster_id = idx_to_cluster[idx]
                if r["coin"] == "ETH" and r["n_packets"] <= SHORT_MAX:
                    include_as_mining = False
                    flag = "eth_short_post_merge_candidate"
                    n_eth_short_flagged += 1

            record_out = {
                "vps_group": vps_group,
                "label": r["label"],
                "coin": r["coin"],
                "include_as_mining": include_as_mining,
                "flag": flag,
                "cluster_id": cluster_id,
                "n_packets_real": r["n_packets"],
                "sizes": sizes_p,
                "directions": dirs_p,
                "interarrival_s": gaps_p,
                "mask": mask,
            }
            out_f.write(json.dumps(record_out) + "\n")
            n_written += 1

            key = "mining" if r["label"] == "mining" and include_as_mining else \
                  ("mining_excluded" if flag else "normal")
            group_label_counts[vps_group][key] += 1

    print(f"\nFlows written: {n_written}")
    print(f"ETH short flows flagged (excluded from main mining class): {n_eth_short_flagged}")
    print("\nPer-VPS counts after flagging (mining / mining_excluded / normal):")
    for ip in sorted(SERVER_IPS):
        c = group_label_counts[ip]
        print(f"{ip:<18} mining={c.get('mining', 0):<4} "
              f"excluded={c.get('mining_excluded', 0):<4} "
              f"normal={c.get('normal', 0)}")

    print(
        "\nCheck the per-VPS mining counts above before finalizing the "
        "GroupKFold split - every VPS needs at least a few mining flows "
        "remaining after the ETH exclusion, or leave-one-VPS-out folds "
        "for that VPS will have zero positives."
    )


if __name__ == "__main__":
    main()
