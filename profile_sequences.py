import csv
import json
from collections import Counter, defaultdict

JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_sequences.jsonl"
RAW_CSV = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\mined_aggregated_data_set.csv"
CHECK_BAD_LEN = True

SERVER_IPS = {
    "157.230.14.71", "157.230.14.73", "157.230.6.255", "157.230.10.203",
    "157.230.14.174", "157.230.14.28", "204.48.26.169", "159.89.85.4",
    "147.182.169.120", "161.35.107.208",
}
SIG_LEN = 16
SHORT_MAX = 31  # "short" = fewer than 32 packets

BUCKETS = [("2", 2, 2), ("3", 3, 3), ("4-7", 4, 7), ("8-15", 8, 15),
           ("16-31", 16, 31), ("32-63", 32, 63), ("64+", 64, 10 ** 12)]


def bucket_label(n):
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return "other"


def profile_jsonl():
    mining_by_server = Counter()
    mining_coin_server = defaultdict(Counter)
    mining_len_by_coin = defaultdict(lambda: [0, 0])  # [short, long]
    normal_by_server = Counter()
    normal_bucket = Counter()
    mid_ports = Counter()
    mid_servers = Counter()
    sigs = {"mining": Counter(), "normal": Counter()}
    n_lines = no_server = 0

    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_lines += 1
            server_is_a = r["ip_a"] in SERVER_IPS
            if not server_is_a and r["ip_b"] not in SERVER_IPS:
                no_server += 1
            server_ip = r["ip_a"] if server_is_a else r["ip_b"]
            n = r["n_packets"]

            # directions: 1 = a->b. Outbound = server -> remote.
            out = [(d == 1) == server_is_a for d in r["directions"][:SIG_LEN]]
            sig = tuple(zip(r["packet_sizes"][:SIG_LEN], out))

            if r["label"] == "mining":
                mining_by_server[server_ip] += 1
                mining_coin_server[r["coin"]][server_ip] += 1
                mining_len_by_coin[r["coin"]][0 if n <= SHORT_MAX else 1] += 1
                sigs["mining"][sig] += 1
            else:
                normal_by_server[server_ip] += 1
                normal_bucket[bucket_label(n)] += 1
                sigs["normal"][sig] += 1
                if 16 <= n <= 31:
                    mid_ports[r["port_a"]] += 1
                    mid_ports[r["port_b"]] += 1
                    mid_servers[server_ip] += 1

    print(f"Lines read: {n_lines} (flows with no server endpoint: {no_server})\n")

    print("=== 1. Flows per VPS (server IP): mining / normal ===")
    for ip in sorted(SERVER_IPS):
        print(f"{ip:<18} mining={mining_by_server.get(ip, 0):<5} "
              f"normal={normal_by_server.get(ip, 0)}")
    print("\nCoin x VPS (mining flows):")
    for coin, per_server in sorted(mining_coin_server.items()):
        print(f"{coin}: {dict(per_server)}")

    print("\n=== 2. Mining flows short(<32 pkts) / long(>=32) by coin ===")
    for coin, (short, long_) in sorted(mining_len_by_coin.items()):
        print(f"{coin}: short={short}, long={long_}")

    print("\n=== 3. Normal-flow packet-count buckets ===")
    total_normal = sum(normal_bucket.values())
    for name, _, _ in BUCKETS:
        c = normal_bucket.get(name, 0)
        print(f"{name:<6} {c:>8} ({100 * c / total_normal:.2f}%)")
    print("\nTop 10 ports inside the normal 16-31 packet bucket (both sides counted):")
    for port, c in mid_ports.most_common(10):
        print(f"port {port}: {c}")
    print("Top 5 VPSs inside that bucket:")
    for ip, c in mid_servers.most_common(5):
        print(f"{ip}: {c}")

    print(f"\n=== 4. Near-duplicate check (first {SIG_LEN} packets: size + direction) ===")
    for name in ("mining", "normal"):
        c = sigs[name]
        total = sum(c.values())
        top5 = sum(v for _, v in c.most_common(5))
        print(f"{name}: {total} flows, {len(c)} distinct signatures, "
              f"top-5 signatures cover {100 * top5 / total:.2f}% of flows")


def show_bad_len(limit=10):
    print("\n=== 5. Rows skipped for non-numeric frame.len ===")
    shown = total = 0
    with open(RAW_CSV, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if not (row.get("ip.src") or "") or not (row.get("ip.dst") or ""):
                continue
            if not (row.get("tcp.srcport") or "") or not (row.get("tcp.dstport") or ""):
                continue
            raw = row.get("frame.len") or ""
            try:
                int(float(raw))
            except ValueError:
                total += 1
                if shown < limit:
                    print(repr(row))
                    shown += 1
    print(f"Total such rows: {total} (build script reported 88)")


if __name__ == "__main__":
    profile_jsonl()
    if CHECK_BAD_LEN:
        show_bad_len()
