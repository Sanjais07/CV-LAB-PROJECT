"""
test_fixed_source_port_hypothesis.py

Hypothesis: ports like 49107, 16234, 40709 etc. that appear thousands of
times in the 'other' bucket are FIXED SOURCE PORTS used by mass-scanning
tools (zmap/masscan commonly do this), not real ephemeral client ports.

Test: for each of the given ports, count how many DISTINCT counterpart
(ip, port) pairs it appears alongside. A fixed scanner source port should
pair with many distinct remote IPs and/or many distinct destination ports.
A real, single persistent connection would show a small, stable set of
counterparts instead. This script only reports the counts - it doesn't
assume the answer.
"""

import csv
from collections import defaultdict

INPUT_CSV = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_dataset_cleaned.csv"

PORTS_TO_TEST = [
    "49107", "16234", "40709", "41831", "48371", "23", "37215",
    "61000", "52018", "6379", "48390",
]


def main():
    # port -> set of (counterpart_ip, counterpart_port)
    counterparts = defaultdict(set)
    flow_count = defaultdict(int)

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] != "normal":
                continue
            ip_a, port_a = row["ip_a"], row["port_a"]
            ip_b, port_b = row["ip_b"], row["port_b"]

            if port_a in PORTS_TO_TEST:
                counterparts[port_a].add((ip_b, port_b))
                flow_count[port_a] += 1
            if port_b in PORTS_TO_TEST:
                counterparts[port_b].add((ip_a, port_a))
                flow_count[port_b] += 1

    print(f"{'Port':<8} {'Flows':>8} {'Distinct counterparts':>22} {'Ratio':>8}")
    for port in PORTS_TO_TEST:
        n_flows = flow_count.get(port, 0)
        n_counterparts = len(counterparts.get(port, set()))
        ratio = n_counterparts / n_flows if n_flows else 0
        print(f"{port:<8} {n_flows:>8} {n_counterparts:>22} {ratio:>8.2f}")

    print(
        "\nRatio near 1.0 = each flow on this port has a near-unique "
        "counterpart (consistent with a scanner hitting many different "
        "targets from one fixed source port). Ratio near 0 (few distinct "
        "counterparts across many flows) = a small number of repeat "
        "connections instead, which would NOT support the scanning "
        "hypothesis for that port."
    )


if __name__ == "__main__":
    main()
