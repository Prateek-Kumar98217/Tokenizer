from collections import Counter
import math
import ast

def _get_pair_counts(freqs: Counter) -> Counter:
    pair_counts = Counter()
    for word, freq in freqs.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
    return pair_counts

def _apply_merge(freqs: Counter, best_pair: tuple, new_idx: int) -> Counter:
    new_freqs = Counter()
    p0, p1 = best_pair
    for word, freq in freqs.items():
        if p0 not in word:
            new_freqs[word] += freq
            continue
        
        new_tuple = []
        i = 0
        while i< len(word):
            if i < len(word) -1 and word[i] == p0 and word[i+1] == p1:
                new_tuple.append(new_idx)
                i += 2
            else:
                new_tuple.append(word[i])
                i += 1
        new_freqs[tuple(new_tuple)] += freq
    return new_freqs

def train(freqs: Counter, target_vocab: int, use_limit: bool = True):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges = {}
    current_idx = 256

    if use_limit:
        limit = int(math.sqrt(freqs.most_common(1)[0][1]))

    print(f"[Trainer] Using frequency limit: {limit}" if use_limit else f"[Trainer] Using target vocab size: {target_vocab}")

    while True:
        if len(vocab) >= target_vocab and not use_limit:
            break
            
        pair_counts = _get_pair_counts(freqs)
        if not pair_counts:
            break

        if use_limit and pair_counts.most_common(1)[0][1] < limit:
            break

        best_pair = pair_counts.most_common(1)[0][0]

        merges[best_pair] = current_idx
        vocab[current_idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        freqs = _apply_merge(freqs, best_pair, current_idx)
        current_idx += 1
        
        if (current_idx - 256) % 100 == 0:
            print(f"Merged {current_idx - 256} pairs. Current vocab size: {current_idx}")
            with open("./artifacts/vocab.txt", "w", encoding="utf-8") as f:
                for idx, token in sorted(vocab.items()):
                    f.write(f"{idx}\t{token.decode('utf-8', errors='backslashreplace')}\n")

            with open("./artifacts/merges.txt", "w", encoding="utf-8") as f:
                for pair, idx in sorted(merges.items(), key=lambda item: item[1]):
                    f.write(f"{idx}\t{pair[0]}\t{pair[1]}\n")
            
    return vocab, merges

if __name__ == "__main__":
    corpus = Counter()
    with open("./corpus_vocab.txt", "r", encoding="utf-8") as f:
        for line in f:
            pair_str, count_str = line.strip().split(": ", 1)
            pair = ast.literal_eval(pair_str)
            
            corpus[pair] = int(count_str)

    vocab, merges = train(corpus, target_vocab=100000)