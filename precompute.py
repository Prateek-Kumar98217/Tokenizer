import math
import regex as re
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

TOKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

def process_single_file(filepath: Path):
    local_counter = Counter()
    with open(filepath, "r", encoding = "utf-8", errors="replace") as f:
        content = f.read()
        tokens = re.findall(TOKEN_PATTERN, content)
        for token in tokens:
            token_bytes = token.encode("utf-8")
            local_counter.update(zip(token_bytes, token_bytes[1:]))
    return local_counter

def precompute_corpus(data_dir: str, num_workers: int = None):      
    corpus_path = Path(data_dir)

    print(f"[Scan] Scanning {corpus_path} for files...")
    files = list(corpus_path.glob("*.txt"))
    print(f"[Scan] Found {len(files)} files in {corpus_path}")

    global_counter = Counter()

    with ProcessPoolExecutor(max_workers=num_workers if num_workers else 1) as executor:
        tasks = {executor.submit(process_single_file, file): file for file in files }

        for task in tqdm(as_completed(tasks), total=len(files), desc="[Processor]Processing files..."):
            try:
                local_counter = task.result()
                global_counter.update(local_counter)
            except Exception as e:
                print(f"[Error] File {tasks[task].name} generated an exception: {e}")
    return global_counter

if __name__ == "__main__":
    corpus_vocab = precompute_corpus("./data", num_workers=4)
    print(f"[Processor] Complete Processing Corpus, found {len(corpus_vocab)} byte pairs.")
    top_tokens = corpus_vocab.most_common(20)
    print(f"[Analysis] Top 20 tokens: {top_tokens}")
    limit = math.sqrt(top_tokens[0][1])
    print(f"[Analysis] Limit: {int(limit)}")
    depth_first_merges = 0
    for pair, count in corpus_vocab.items():
        if count > limit:
            depth_first_merges += 1
    print(f"[Analysis] Possible Merges: {depth_first_merges}")
    
    with open("./corpus_vocab.txt", "w", encoding="utf-8") as f:
        for pair, count in corpus_vocab.items():
            f.write(f"{pair}: {count}\n")
