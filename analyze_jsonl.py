import json

def process_jsonl(path):
    total = 0
    mining = 0
    normal = 0
    
    coin_counts = {}
    
    mining_counts = []
    normal_counts = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            data = json.loads(line)
            
            label = data['label']
            n_packets = data['n_packets']
            
            if label == 'mining':
                mining += 1
                coin = data['coin']
                coin_counts[coin] = coin_counts.get(coin, 0) + 1
                mining_counts.append(n_packets)
            elif label == 'normal':
                normal += 1
                normal_counts.append(n_packets)
                
    print(f"1. Total lines: {total}, Mining: {mining}, Normal: {normal}")
    print(f"2. Mining flow count per coin: {coin_counts}")
    
    def pct(counts, threshold):
        if not counts: return 0.0
        return sum(1 for c in counts if c >= threshold) / len(counts) * 100
        
    print(f"3. Mining flows % with n_packets:")
    print(f"   >= 16:  {pct(mining_counts, 16):.2f}%")
    print(f"   >= 32:  {pct(mining_counts, 32):.2f}%")
    print(f"   >= 64:  {pct(mining_counts, 64):.2f}%")
    print(f"   >= 128: {pct(mining_counts, 128):.2f}%")
    
    print(f"4. Normal flows % with n_packets:")
    print(f"   >= 16:  {pct(normal_counts, 16):.2f}%")
    print(f"   >= 32:  {pct(normal_counts, 32):.2f}%")
    print(f"   >= 64:  {pct(normal_counts, 64):.2f}%")
    print(f"   >= 128: {pct(normal_counts, 128):.2f}%")

if __name__ == "__main__":
    process_jsonl(r"D:\9 Sem\CV Proj 2\CSV_MINERS\CSV_MINERS\flow_sequences.jsonl")
