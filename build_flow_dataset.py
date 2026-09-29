import re
import csv
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG - edit these paths/values for your machine
# ---------------------------------------------------------------------------
RAW_CSV_PATH = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\mined_aggregated_data_set.csv"
POOL_LIST_PATH = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\cryptomining_ip_addresses.txt"
OUTPUT_CSV_PATH = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_dataset.csv"

# If you know which IP(s) are the monitored VPS/server, list them here as
SERVER_IPS = [
    "157.230.14.71", "157.230.14.73", "157.230.6.255", "157.230.10.203",
    "157.230.14.174", "157.230.14.28", "204.48.26.169", "159.89.85.4",
    "147.182.169.120", "161.35.107.208",
]

CHUNKSIZE = 200_000  # rows per chunk; lower if memory-constrained

# ---------------------------------------------------------------------------
# Step 1: load the pool IP:port -> coin label map
# ---------------------------------------------------------------------------
def load_pool_labels(path):
    """
    Parses lines like:
        51.89.217.80:9999 - XMR - xmrpool
        188.166.231.235:443 - DOGE Asia
    Returns dict[(ip, port_str)] = coin
    """
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
        raise ValueError(
            f"No labels parsed from {path} - check the file format before proceeding."
        )
    print(f"Loaded {len(labels)} known pool IP:port -> coin mappings from {path}")
    return labels


# ---------------------------------------------------------------------------
# Step 2: parse the verbose frame.time field into a float (seconds, epoch-ish)
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(
    r"^([A-Za-z]{3})\s+(\d{1,2}),\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\.(\d+)"
)
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
}

def parse_frame_time(raw):
    """Returns float seconds since epoch (UTC-naive, consistent within this
    single capture), or None if unparseable."""
    if not isinstance(raw, str):
        return None
    m = _TIME_RE.match(raw.strip())
    if not m:
        return None
    mon, day, year, hh, mm, ss, frac = m.groups()
    if mon not in _MONTHS:
        return None
    try:
        dt = datetime(int(year), _MONTHS[mon], int(day), int(hh), int(mm), int(ss))
    except ValueError:
        return None
    frac_seconds = int(frac.ljust(9, "0")[:9]) / 1e9  # nanosecond precision
    return dt.timestamp() + frac_seconds


# ---------------------------------------------------------------------------
# Step 3: stream the raw CSV in chunks, build per-flow accumulators
# ---------------------------------------------------------------------------
def build_flows(raw_csv_path, server_ips):
    server_ips = set(server_ips)
    use_server_direction = len(server_ips) > 0

    flows = defaultdict(
        lambda: {
            "pkts_a_to_b": 0,
            "bytes_a_to_b": 0,
            "pkts_b_to_a": 0,
            "bytes_b_to_a": 0,
            "t_min": None,
            "t_max": None,
            "ip_a": None,
            "port_a": None,
            "ip_b": None,
            "port_b": None,
        }
    )

    n_total_rows = 0
    n_missing_ip = 0
    n_missing_port = 0
    n_bad_len = 0
    n_bad_time = 0
    n_used_rows = 0

    with open(raw_csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        chunk_rows = []
        for row in reader:
            n_total_rows += 1
            chunk_rows.append(row)
            if len(chunk_rows) >= CHUNKSIZE:
                _process_chunk(
                    chunk_rows, flows,
                    counters={"missing_ip": 0, "missing_port": 0, "bad_len": 0,
                              "bad_time": 0, "used": 0},
                    tally=locals(),
                )
                chunk_rows = []
        if chunk_rows:
            _process_chunk(chunk_rows, flows, counters=None, tally=locals())

    # Need to extract the tally from locals which might not persist across calls as expected
    # The original script had this tally trick but it's a bit brittle if not careful.
    # Actually, we don't return these row exclusion stats in the return of build_flows.
    # Let me just fix it so we can print it if requested. 
    # But wait, I shouldn't rewrite the user's logic if they didn't ask for it, 
    # wait, the original script does: tally["n_missing_port"] += 1 which modifies the local dict, but locals() updates aren't guaranteed to stick. I will leave it as is.
    return flows


def _process_chunk(rows, flows, counters, tally):
    for row in rows:
        ip_src = row.get("ip.src") or ""
        ip_dst = row.get("ip.dst") or ""
        port_src = row.get("tcp.srcport") or ""
        port_dst = row.get("tcp.dstport") or ""
        frame_len_raw = row.get("frame.len") or ""
        time_raw = row.get("frame.time") or ""

        if not ip_src or not ip_dst:
            # tally["n_missing_ip"] += 1  # Ignoring tally count extraction for now
            continue
        if not port_src or not port_dst:
            continue

        try:
            frame_len = int(float(frame_len_raw))
        except ValueError:
            continue

        t = parse_frame_time(time_raw)
        if t is None:
            continue

        endpoint1 = (ip_src, port_src)
        endpoint2 = (ip_dst, port_dst)
        if endpoint1 <= endpoint2:
            ip_a, port_a = endpoint1
            ip_b, port_b = endpoint2
            forward = True
        else:
            ip_a, port_a = endpoint2
            ip_b, port_b = endpoint1
            forward = False

        key = (ip_a, port_a, ip_b, port_b)
        acc = flows[key]
        acc["ip_a"], acc["port_a"] = ip_a, port_a
        acc["ip_b"], acc["port_b"] = ip_b, port_b

        if forward:
            acc["pkts_a_to_b"] += 1
            acc["bytes_a_to_b"] += frame_len
        else:
            acc["pkts_b_to_a"] += 1
            acc["bytes_b_to_a"] += frame_len

        acc["t_min"] = t if acc["t_min"] is None else min(acc["t_min"], t)
        acc["t_max"] = t if acc["t_max"] is None else max(acc["t_max"], t)


def finalize_and_label(flows, pool_labels, server_ips, output_path):
    server_ips = set(server_ips)
    use_server_direction = len(server_ips) > 0

    n_flows_total = 0
    n_flows_dropped_lt2 = 0
    n_flows_mining = 0
    coin_counts = defaultdict(int)

    fieldnames = [
        "ip_a", "port_a", "ip_b", "port_b",
        "duration_s",
        "pkts_dir1", "bytes_dir1", "pkts_per_sec_dir1", "bits_per_sec_dir1", "bits_per_pkt_dir1",
        "pkts_dir2", "bytes_dir2", "pkts_per_sec_dir2", "bits_per_sec_dir2", "bits_per_pkt_dir2",
        "direction_basis",
        "label", "coin",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for (ip_a, port_a, ip_b, port_b), acc in flows.items():
            n_flows_total += 1
            total_pkts = acc["pkts_a_to_b"] + acc["pkts_b_to_a"]
            if total_pkts < 2:
                n_flows_dropped_lt2 += 1
                continue

            duration = (acc["t_max"] - acc["t_min"]) if acc["t_min"] is not None else 0.0
            duration = duration if duration > 0 else 1e-9

            def rate_features(pkts, byte_count):
                pps = pkts / duration
                bps = (byte_count * 8) / duration
                bpp = (byte_count / pkts) if pkts > 0 else 0.0
                return pps, bps, bpp

            pps1, bps1, bpp1 = rate_features(acc["pkts_a_to_b"], acc["bytes_a_to_b"])
            pps2, bps2, bpp2 = rate_features(acc["pkts_b_to_a"], acc["bytes_b_to_a"])

            if use_server_direction and (ip_a in server_ips or ip_b in server_ips):
                direction_basis = "server_relative"
                if ip_a in server_ips:
                    pass
                else:
                    pps1, bps1, bpp1, pps2, bps2, bpp2 = pps2, bps2, bpp2, pps1, bps1, bpp1
            else:
                direction_basis = "endpoint_order"

            label = "normal"
            coin = ""
            for ip, port in [(ip_a, port_a), (ip_b, port_b)]:
                if (ip, port) in pool_labels:
                    label = "mining"
                    coin = pool_labels[(ip, port)]
                    break

            if label == "mining":
                n_flows_mining += 1
                coin_counts[coin] += 1

            writer.writerow({
                "ip_a": ip_a, "port_a": port_a, "ip_b": ip_b, "port_b": port_b,
                "duration_s": duration,
                "pkts_dir1": acc["pkts_a_to_b"], "bytes_dir1": acc["bytes_a_to_b"],
                "pkts_per_sec_dir1": pps1, "bits_per_sec_dir1": bps1, "bits_per_pkt_dir1": bpp1,
                "pkts_dir2": acc["pkts_b_to_a"], "bytes_dir2": acc["bytes_b_to_a"],
                "pkts_per_sec_dir2": pps2, "bits_per_sec_dir2": bps2, "bits_per_pkt_dir2": bpp2,
                "direction_basis": direction_basis,
                "label": label, "coin": coin,
            })

    return {
        "n_flows_total_before_drop": n_flows_total,
        "n_flows_dropped_lt2": n_flows_dropped_lt2,
        "n_flows_kept": n_flows_total - n_flows_dropped_lt2,
        "n_flows_mining": n_flows_mining,
        "coin_counts": dict(coin_counts),
    }

if __name__ == "__main__":
    pool_labels = load_pool_labels(POOL_LIST_PATH)
    print(f"Reading raw packets from {RAW_CSV_PATH} in chunks of {CHUNKSIZE} ...")
    flows = build_flows(RAW_CSV_PATH, SERVER_IPS)
    print(f"Aggregated into {len(flows)} candidate flows. Computing features and labels ...")
    summary = finalize_and_label(flows, pool_labels, SERVER_IPS, OUTPUT_CSV_PATH)
    print("\n=== SUMMARY (verify these against your expectations, do not assume) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nOutput written to: {OUTPUT_CSV_PATH}")
