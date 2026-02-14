# reranker.py
from sentence_transformers import CrossEncoder
class ReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"):
        self.ce = CrossEncoder(model_name, device=device)

    def rerank(self, query, candidates: list):
        # candidates: list of dicts with "text"
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.ce.predict(pairs)
        for s,c in zip(scores, candidates):
            c["score_rerank"] = float(s)
        candidates.sort(key=lambda x: x["score_rerank"], reverse=True)
        return candidates
