import re
import csv
import json
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_CSV_PATH = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\mined_aggregated_data_set.csv"
POOL_LIST_PATH = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\cryptomining_ip_addresses.txt"
OUTPUT_JSONL_PATH = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_sequences.jsonl"

SERVER_IPS = {
    "157.230.14.71", "157.230.14.73", "157.230.6.255", "157.230.10.203",
    "157.230.14.174", "157.230.14.28", "204.48.26.169", "159.89.85.4",
    "147.182.169.120", "161.35.107.208",
}


# ---------------------------------------------------------------------------
# Pool labels (same logic as build_flow_dataset.py)
# ---------------------------------------------------------------------------
def load_pool_labels(path):
    labels = {}
    line_re = re.compile(r"^\s*([\d.]+):(\d+)\s*-\s*([A-Za-z]+)")
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            m = line_re.match(line)
            if not m:
                print(f"WARNING: could not parse pool-list line, skipping: {line!r}")
                continue
            ip, port, coin = m.group(1), m.group(2), m.group(3).upper()
            labels[(ip, port)] = coin
    if not labels:
        raise ValueError(f"No labels parsed from {path}")
    print(f"Loaded {len(labels)} known pool IP:port -> coin mappings")
    return labels


# ---------------------------------------------------------------------------
# frame.time parsing (same logic as build_flow_dataset.py)
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(
    r"^([A-Za-z]{3})\s+(\d{1,2}),\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\.(\d+)"
)
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
}


def parse_frame_time(raw):
    if not isinstance(raw, str):
        return None
    m = _TIME_RE.match(raw.strip())
    if not m:
        return None
    mon, day, year, hh, mm, ss, frac = m.groups()
    if mon not in _MONTHS:
        return None
    try:
        dt = datetime(int(year), _MONTHS[mon], int(day),
                      int(hh), int(mm), int(ss))
    except ValueError:
        return None
    frac_seconds = int(frac.ljust(9, "0")[:9]) / 1e9
    return dt.timestamp() + frac_seconds


# ---------------------------------------------------------------------------
# Step 1: stream raw CSV, collect (time, frame_len, forward) per flow key
# ---------------------------------------------------------------------------
def build_flows(raw_csv_path):
    flows = defaultdict(list)  # key -> [(t, frame_len, forward), ...]

    n_total = n_used = 0
    n_missing_ip = n_missing_port = n_bad_len = n_bad_time = 0

    with open(raw_csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            ip_src = row.get("ip.src") or ""
            ip_dst = row.get("ip.dst") or ""
            port_src = row.get("tcp.srcport") or ""
            port_dst = row.get("tcp.dstport") or ""
            frame_len_raw = row.get("frame.len") or ""
            time_raw = row.get("frame.time") or ""

            if not ip_src or not ip_dst:
                n_missing_ip += 1
                continue
            if not port_src or not port_dst:
                n_missing_port += 1
                continue

            try:
                frame_len = int(float(frame_len_raw))
            except ValueError:
                n_bad_len += 1
                continue

            t = parse_frame_time(time_raw)
            if t is None:
                n_bad_time += 1
                continue

            endpoint1 = (ip_src, port_src)
            endpoint2 = (ip_dst, port_dst)
            if endpoint1 <= endpoint2:
                key = endpoint1 + endpoint2      # (ip_a, port_a, ip_b, port_b)
                forward = True
            else:
                key = endpoint2 + endpoint1
                forward = False

            flows[key].append((t, frame_len, forward))
            n_used += 1

            if n_total % 500_000 == 0:
                print(f"  ...{n_total} raw rows read, {len(flows)} flow keys so far")

    print(f"\nRaw rows: total={n_total}, used={n_used}, "
          f"missing_ip={n_missing_ip}, missing_port(non-TCP)={n_missing_port}, "
          f"bad_len={n_bad_len}, bad_time={n_bad_time}")
    return flows


# ---------------------------------------------------------------------------
# Step 2: sort, filter, compute inter-arrival times, label, write JSONL
# ---------------------------------------------------------------------------
def finalize_and_write(flows, pool_labels, output_path):
    n_seen = n_lt2 = n_vps_to_vps = n_written = n_mining = 0
    counts = {"mining": [], "normal": []}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for (ip_a, port_a, ip_b, port_b), packets in flows.items():
            n_seen += 1

            if len(packets) < 2:
                n_lt2 += 1
                continue
            if ip_a in SERVER_IPS and ip_b in SERVER_IPS:
                n_vps_to_vps += 1
                continue

            packets.sort(key=lambda p: p[0])

            sizes, directions, gaps = [], [], []
            prev_t = None
            for t, size, forward in packets:
                sizes.append(size)
                directions.append(1 if forward else 0)
                gaps.append(0.0 if prev_t is None else t - prev_t)
                prev_t = t

            label, coin = "normal", ""
            for ip, port in ((ip_a, port_a), (ip_b, port_b)):
                if (ip, port) in pool_labels:
                    label, coin = "mining", pool_labels[(ip, port)]
                    break

            if label == "mining":
                n_mining += 1
            counts[label].append(len(sizes))

            out_f.write(json.dumps({
                "ip_a": ip_a, "port_a": port_a, "ip_b": ip_b, "port_b": port_b,
                "label": label, "coin": coin,
                "n_packets": len(sizes),
                "packet_sizes": sizes,
                "directions": directions,
                "interarrival_s": gaps,
            }) + "\n")
            n_written += 1

    def summarize(vals):
        if not vals:
            return "n=0"
        vals = sorted(vals)
        n = len(vals)
        return (f"n={n}, min={vals[0]}, median={vals[n // 2]}, "
                f"max={vals[-1]}, mean={sum(vals) / n:.1f}")

    print(f"\nFlow keys seen: {n_seen}")
    print(f"Dropped (<2 packets): {n_lt2}")
    print(f"Dropped (VPS-to-VPS admin): {n_vps_to_vps}")
    print(f"Flows written: {n_written}")
    print(f"Mining flows: {n_mining}")
    print(f"Packet counts, mining flows: {summarize(counts['mining'])}")
    print(f"Packet counts, normal flows: {summarize(counts['normal'])}")
    print("\nUse the mining packet-count distribution to choose your sequence "
          "window length: a window longer than most mining flows means "
          "mostly padding.")


if __name__ == "__main__":
    pool_labels = load_pool_labels(POOL_LIST_PATH)
    print(f"Reading raw packets from {RAW_CSV_PATH} ...")
    flows = build_flows(RAW_CSV_PATH)
    print(f"\nGrouped into {len(flows)} candidate flow keys. Finalizing ...")
    finalize_and_write(flows, pool_labels, OUTPUT_JSONL_PATH)
    print(f"\nOutput written to: {OUTPUT_JSONL_PATH}")
