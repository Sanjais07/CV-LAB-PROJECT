"""
profile_other_bucket.py

The composition report showed 58.79% of the 'normal' class (139,155 flows)
falling into "other/unclassified" - i.e. neither side of the flow was on
port 22/53/80/443/8080. This script only reports the actual port frequency
distribution within that bucket - it draws no conclusion about what the
traffic "is", since that requires looking at the real numbers first.
"""

import csv
from collections import Counter

INPUT_CSV = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_dataset_cleaned.csv"

KNOWN_BUCKETED_PORTS = {"22", "53", "80", "443", "8080"}


def main():
    port_counter = Counter()
    n_other_flows = 0
    unique_ports_seen = set()
    single_hit_ports = 0  # ports that appear in exactly 1 flow

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] != "normal":
                continue
            port_a, port_b = row["port_a"], row["port_b"]
            if port_a in KNOWN_BUCKETED_PORTS or port_b in KNOWN_BUCKETED_PORTS:
                continue  # already accounted for in the previous report

            n_other_flows += 1
            # Count both ports seen on this flow (the non-ephemeral one is
            # usually the meaningful one, but report both - don't guess
            # which side is ephemeral).
            port_counter[port_a] += 1
            port_counter[port_b] += 1
            unique_ports_seen.add(port_a)
            unique_ports_seen.add(port_b)

    print(f"Total flows in 'other/unclassified' bucket: {n_other_flows}")
    print(f"Distinct ports appearing in this bucket: {len(unique_ports_seen)}")
    print()

    for port, count in port_counter.items():
        if count == 1:
            single_hit_ports += 1
    pct_single_hit = 100 * single_hit_ports / len(unique_ports_seen) if unique_ports_seen else 0
    print(f"Ports appearing exactly once: {single_hit_ports} "
          f"({pct_single_hit:.2f}% of distinct ports)")
    print(
        "(A high percentage here is consistent with internet-wide scanning: "
        "lots of distinct destination ports each hit once. A low percentage "
        "with a few dominant ports below instead means real clustered "
        "services you should name explicitly.)"
    )
    print()

    print("=== Top 25 ports by flow count in the 'other/unclassified' bucket ===")
    for port, count in port_counter.most_common(25):
        pct = 100 * count / n_other_flows if n_other_flows else 0
        print(f"port {port}: {count} flows ({pct:.2f}% of this bucket)")


if __name__ == "__main__":
    main()
