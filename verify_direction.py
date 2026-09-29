import csv

FLOW_CSV_PATH = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_dataset.csv"
SERVER_IPS = [
    "157.230.14.71", "157.230.14.73", "157.230.6.255", "157.230.10.203",
    "157.230.14.174", "157.230.14.28", "204.48.26.169", "159.89.85.4",
    "147.182.169.120", "161.35.107.208",
]

server_relative_count = 0
endpoint_order_count = 0
endpoint_order_examples = []

vps_to_vps_flows = []

with open(FLOW_CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        db = row["direction_basis"]
        if db == "server_relative":
            server_relative_count += 1
        elif db == "endpoint_order":
            endpoint_order_count += 1
            if len(endpoint_order_examples) < 5:
                endpoint_order_examples.append(row)
        
        if row["ip_a"] in SERVER_IPS and row["ip_b"] in SERVER_IPS:
            vps_to_vps_flows.append(row)

print("--- STEP 3: Verification ---")
print(f"direction_basis == 'server_relative': {server_relative_count}")
print(f"direction_basis == 'endpoint_order': {endpoint_order_count}")
print(f"\n5 Examples of 'endpoint_order' flows (out of {endpoint_order_count}):")
for ex in endpoint_order_examples:
    print(f"{ex['ip_a']}:{ex['port_a']} <-> {ex['ip_b']}:{ex['port_b']} | duration: {ex['duration_s']} | label: {ex['label']}")

print("\n--- STEP 4: VPS-to-VPS Flows ---")
print(f"Total VPS-to-VPS flows: {len(vps_to_vps_flows)}")
for flow in vps_to_vps_flows:
    print(f"{flow['ip_a']}:{flow['port_a']} <-> {flow['ip_b']}:{flow['port_b']} | label: {flow['label']}")
