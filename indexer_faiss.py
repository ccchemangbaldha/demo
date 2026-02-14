# indexer_faiss.py
import faiss, numpy as np
from pathlib import Path

class FaissIndex:
    def __init__(self, dim, path=None):
        self.dim = dim
        self.path = path or "faiss.index"
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        # load if exists
        p = Path(self.path)
        if p.exists():
            self.index = faiss.read_index(str(p))

    def add(self, ids, vectors):
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)
        self.index.add_with_ids(vectors, ids)

    def search(self, vector, topk=10):
        v = vector.astype('float32')
        faiss.normalize_L2(v)
        D, I = self.index.search(v, topk)
        return D, I

    def save(self):
        faiss.write_index(self.index, self.path)
