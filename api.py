# api.py
import os, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from indexer_elastic import ElasticIndexer
from indexer_faiss import FaissIndex
from reranker import ReRanker
from utils import load_docstore
import numpy as np
from embedder import BGEEmbedder

load_dotenv()
ELASTIC_URL = os.getenv("ELASTIC_URL")
ELASTIC_USER = os.getenv("ELASTIC_USER")
ELASTIC_PASS = os.getenv("ELASTIC_PASS")
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX","maths_texts")
FAISS_PATH = os.getenv("FAISS_INDEX_PATH","faiss.index")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH","docstore.json")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL","BAAI/bge-base-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2")

elastic = ElasticIndexer(ELASTIC_URL, ELASTIC_USER, ELASTIC_PASS, ELASTIC_INDEX)
faiss = FaissIndex(dim=1536, path=FAISS_PATH)  # dim should match embedder; replace if different
embedder = BGEEmbedder(EMBED_MODEL)
reranker = ReRanker(RERANKER_MODEL)
docstore = load_docstore(DOCSTORE_PATH)

app = FastAPI()

class Query(BaseModel):
    query: str
    subject: str = "maths"
    chapter: int = None
    section: str = None
    top_k: int = 5

@app.get("/")
def root():
    return {"message":"hello from hemang"}

@app.post("/search")
def search(q: Query):
    # 1) quick metadata match: if user asked "example 6 chapter 2" try filter
    filters={}
    if q.chapter:
        filters["chapter"] = q.chapter
    if q.section:
        filters["section"] = q.section

    # 2) BM25 search
    bm25 = elastic.search_bm25(q.query, size=30, filters=filters if filters else None)
    bm25_ids = [int(x[0]) for x in bm25]

    # 3) vector search
    vec = embedder.embed_texts([q.query])
    D, I = faiss.search(vec, topk=30)
    faiss_ids = [int(i) for i in I[0] if i != -1]

    # 4) merge candidates (unique), fetch from elastic
    candidate_ids = list(dict.fromkeys(bm25_ids + faiss_ids))[:50]
    if not candidate_ids:
        return {"results":[]}
    docs = elastic.get_docs(candidate_ids)
    # normalize into list of dicts
    candidates = []
    for d in docs:
        candidates.append({"id": int(d["_id"]),"text": d["_source"]["text"], "meta": d["_source"]})

    # 5) rerank
    ranked = reranker.rerank(q.query, candidates)[:q.top_k]
    return {"results": ranked}
