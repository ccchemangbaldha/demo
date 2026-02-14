# chunker.py
from typing import List
import math
from itertools import chain
import uuid

def chunk_blocks(blocks, max_tokens=700, overlap=120):
    """
    blocks: list of dicts from parser
    returns list of chunks with preserved metadata
    """
    chunks = []
    for b in blocks:
        text = b["text"].strip()
        if not text:
            continue
        # don't split examples/exercise/proof
        if b["type"].startswith("example") or b["type"].startswith("exercise") or "proof" in b["type"].lower():
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "meta": {k:v for k,v in b.items() if k!="text"}
            })
            continue

        # token-estimation by words ~ 1 token ≈ 0.75 words; to keep simple, use words count
        words = text.split()
        approx_tokens = len(words)
        if approx_tokens <= max_tokens:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "meta": {k:v for k,v in b.items() if k!="text"}
            })
            continue

        # split into word windows (sliding)
        step = max_tokens - overlap
        start = 0
        while start < len(words):
            seg = words[start:start+max_tokens]
            seg_text = " ".join(seg)
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": seg_text,
                "meta": {**{k:v for k,v in b.items() if k!="text"}, "chunk_start_word": start}
            })
            start += step
    return chunks
