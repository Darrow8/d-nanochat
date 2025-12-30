
import regex as re
from collections import Counter, defaultdict
import time
import rustbpe
import tiktoken
import pytest

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Reference tokenizer, pretty much copy pasted and pruned a bit from minbpe

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


rustbpe_tokenizer = rustbpe.Tokenizer()

# step 1: train tokenizer
with open("../../shakespeare/romeo-and-juliet_TXT_FolgerShakespeare.txt", "r", encoding="utf-8") as f:
    shakespeare = f.read()
    shakespeare_arr = shakespeare.splitlines()
    # Get all distinct characters in shakespeare
    char_set = set("".join(shakespeare_arr))
    vocab_size = len(char_set)
    print(f"Number of distinct characters (vocab size): {vocab_size}")

    print(f"Got Romeo & Juliet {"\n".join(shakespeare_arr[:2])}")

rustbpe_tokenizer.train_from_iterator(shakespeare_arr, 512)  # 512 - 256 = 256 merges

input_text = "The unexamined life is not worth living"

print(f"input_text: {input_text}")

encoded_text = rustbpe_tokenizer.encode(input_text)

print(f"encoded: {encoded_text}")

decoded_text = rustbpe_tokenizer.decode(encoded_text)

print(f"decoded: {decoded_text}")

merges_hashmap = rustbpe_tokenizer.merges

# print(f"merges_hashmap {merges_hashmap}")