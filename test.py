import regex as re

testfile = "something.txt"
vocab = {idx: bytes([idx]) for idx in range(256)}
merge = {}
with open(testfile, "r", encoding="utf-8") as f:
    content = f.read()
    
token_split_regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
content_split = re.findall(token_split_regex, content)

ids = [list(ch.encode("utf-8")) for ch in content_split]
merge_num = 1
while True:
    counts = {}
    for chunk in ids:
        for pair in zip(chunk, chunk[1:]):
            counts[pair] = counts.get(pair, 0) + 1
            
    if not counts:
        break
        
    pair = max(counts, key=counts.get)
    new_id = 256 + merge_num
    
    new_ids = []
    for chunk in ids:
        i = 0
        new_chunk = []
        while i < len(chunk):
            if chunk[i] == pair[0] and i < len(chunk) - 1 and chunk[i+1] == pair[1]:
                new_chunk.append(new_id)
                i += 2
            else:
                new_chunk.append(chunk[i])
                i += 1
        new_ids.append(new_chunk)
        
    ids = new_ids 
    
    merge[new_id] = pair
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    
    if merge_num > 250:
        break
    merge_num += 1

with open("vocab.txt", "w", encoding="utf-8") as f:
    for idx, token in sorted(vocab.items()):
        f.write(f"{idx}\t{token.decode('utf-8', errors='backslashreplace')}\n")

with open("merge.txt", "w", encoding="utf-8") as f:
    for idx, pair in sorted(merge.items()):
        f.write(f"{idx}\t{pair[0]}\t{pair[1]}\n")
        
print(len(content_split))
print(len(vocab))
print(len(merge))