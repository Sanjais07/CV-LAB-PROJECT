import csv
from collections import defaultdict, Counter

FLOW_CSV_PATH = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_dataset.csv"

PAPER_TOTAL_FLOWS = 84344
PAPER_NORMAL_FLOWS = 84187


def main():
    fan_in = defaultdict(set)   # ip -> set of distinct counterpart ips
    port_counter = Counter()    # port -> count of flows touching that port (either side)
    total_flows = 0
    port22_flows = 0
    label_counts = Counter()

    with open(FLOW_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_flows += 1
            ip_a, port_a = row["ip_a"], row["port_a"]
            ip_b, port_b = row["ip_b"], row["port_b"]
            label_counts[row["label"]] += 1

            fan_in[ip_a].add(ip_b)
            fan_in[ip_b].add(ip_a)

            port_counter[port_a] += 1
            port_counter[port_b] += 1

            if port_a == "22" or port_b == "22":
                port22_flows += 1

    print(f"Total flows read: {total_flows}")
    print(f"Label breakdown: {dict(label_counts)}")
    print()

    # --- H1: fan-in / server-candidate check ---
    print("=== H1 check: candidate server IPs (top 20 by distinct counterpart count) ===")
    ranked = sorted(fan_in.items(), key=lambda kv: len(kv[1]), reverse=True)
    for ip, counterparts in ranked[:20]:
        print(f"{ip}: talks to {len(counterparts)} distinct other IPs")
    print(
        "\nIf ~10 IPs stand out sharply above the rest here, that supports "
        "the 'this is a merged multi-VPS capture' hypothesis. If it's a "
        "smooth long tail instead, it doesn't."
    )
    print()

    # --- H2: SSH / admin traffic share ---
    print("=== H2 check: how much of the total is SSH (port 22) traffic ===")
    excess = total_flows - PAPER_TOTAL_FLOWS
    print(f"Flows touching port 22 (SSH): {port22_flows} "
          f"({100 * port22_flows / total_flows:.2f}% of all flows)")
    print(f"Flows in excess of the paper's reported total ({PAPER_TOTAL_FLOWS}): {excess}")
    if excess > 0:
        print(f"Port-22 flows as a share of that excess: "
              f"{100 * min(port22_flows, excess) / excess:.2f}% (rough upper bound)")
    print()

    # --- General port composition, for context ---
    print("=== Top 15 ports overall (either side of the flow) ===")
    for port, count in port_counter.most_common(15):
        print(f"port {port}: {count} flows")


if __name__ == "__main__":
    main()
