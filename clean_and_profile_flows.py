import csv
from collections import Counter

INPUT_CSV = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_dataset.csv"
OUTPUT_CSV = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_dataset_cleaned.csv"

SERVER_IPS = {
    "157.230.14.71", "157.230.14.73", "157.230.6.255", "157.230.10.203",
    "157.230.14.174", "157.230.14.28", "204.48.26.169", "159.89.85.4",
    "147.182.169.120", "161.35.107.208",
}

# Well-known port buckets for reporting only - does not affect filtering.
KNOWN_PORT_NAMES = {
    "22": "SSH",
    "53": "DNS/53 (likely scan noise on TCP)",
    "80": "HTTP",
    "443": "HTTPS/TLS",
    "8080": "HTTP-alt",
}


def main():
    n_read = 0
    n_dropped_vps_to_vps = 0
    n_written = 0
    label_counts = Counter()
    normal_port_bucket = Counter()

    with open(INPUT_CSV, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            n_read += 1
            ip_a, ip_b = row["ip_a"], row["ip_b"]

            if ip_a in SERVER_IPS and ip_b in SERVER_IPS:
                n_dropped_vps_to_vps += 1
                continue

            writer.writerow(row)
            n_written += 1
            label_counts[row["label"]] += 1

            if row["label"] == "normal":
                port_a, port_b = row["port_a"], row["port_b"]
                bucket = None
                for p in (port_a, port_b):
                    if p in KNOWN_PORT_NAMES:
                        bucket = KNOWN_PORT_NAMES[p]
                        break
                normal_port_bucket[bucket or "other/unclassified"] += 1

    print(f"Rows read: {n_read}")
    print(f"Dropped (VPS-to-VPS admin traffic): {n_dropped_vps_to_vps}")
    print(f"Rows written to {OUTPUT_CSV}: {n_written}")
    print(f"Label counts after cleaning: {dict(label_counts)}")
    print()
    print("=== Composition of the 'normal' class by port bucket ===")
    n_normal = label_counts["normal"]
    for bucket, count in normal_port_bucket.most_common():
        pct = 100 * count / n_normal if n_normal else 0
        print(f"{bucket}: {count} flows ({pct:.2f}% of normal class)")
    print()
    print(
        "This table is your evidence for the methodology section: state "
        "explicitly that the normal class includes real SSH admin traffic "
        "and real internet background scan noise, by design, as a more "
        "realistic negative class than an idle-traffic baseline."
    )


if __name__ == "__main__":
    main()
