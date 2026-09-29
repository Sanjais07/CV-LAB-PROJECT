import json
from collections import Counter

OUTPUT_JSONL = r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\modeling_dataset.jsonl"

def main():
    total_lines = 0
    mining_included = 0
    cluster_counts = Counter()
    
    first_5 = []
    
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            r = json.loads(line)
            
            if total_lines <= 5:
                first_5.append(r)
                
            if r["label"] == "mining" and r["include_as_mining"] == True:
                mining_included += 1
                cluster_counts[r["cluster_id"]] += 1
                
    print(f"1. Total lines: {total_lines}, Mining (included) size: {mining_included}")
    
    distinct_clusters = len(cluster_counts)
    top_3 = cluster_counts.most_common(3)
    
    print(f"2. Distinct cluster_id values remaining: {distinct_clusters}")
    print(f"   Size of 3 largest clusters: {top_3}")
    
    print(f"3. First 5 records padding sanity check:")
    for i, r in enumerate(first_5):
        mask = r["mask"]
        sizes = r["sizes"]
        directions = r["directions"]
        interarrival = r["interarrival_s"]
        
        mask_ok = all(m in [0, 1] for m in mask)
        len_sizes = len(sizes)
        len_dirs = len(directions)
        len_inter = len(interarrival)
        len_mask = len(mask)
        
        ok = mask_ok and (len_sizes == len_dirs == len_inter == len_mask == 64)
        
        print(f"   Record {i+1}: Mask only 0/1: {mask_ok}, Lengths (sizes={len_sizes}, dirs={len_dirs}, gaps={len_inter}, mask={len_mask}). Valid: {ok}")

if __name__ == "__main__":
    main()
