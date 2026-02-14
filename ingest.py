# ingest.py
import os
import numpy as np
from dotenv import load_dotenv
from parser import parse_pdf
from chunker import chunk_blocks
from embedder import BGEEmbedder
from indexer_faiss import FaissIndex
from indexer_elastic import ElasticIndexer
from utils import uid, save_docstore, load_docstore

load_dotenv()

ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "maths_chunks")

FAISS_PATH = os.getenv("FAISS_INDEX_PATH", "faiss.index")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "docstore.json")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))


def main(pdf_path):
    print("Parsing PDF...")
    blocks = parse_pdf(pdf_path)
    print(f"Parsed {len(blocks)} blocks.")

    print("Chunking blocks...")
    chunks = chunk_blocks(blocks)
    print(f"Produced {len(chunks)} chunks.")

    # Elastic Cloud connection
    elastic = ElasticIndexer(
        cloud_id=ELASTIC_CLOUD_ID or "",
        api_key=ELASTIC_API_KEY or "",
        index=ELASTIC_INDEX
    )

    # Embedding model
    embedder = BGEEmbedder(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    ids = [uid() for _ in chunks]

    print("Generating embeddings...")
    vectors = embedder.embed_texts(texts, batch_size=BATCH_SIZE)

    # FAISS index
    print("Saving vectors to FAISS...")
    faiss_index = FaissIndex(dim=vectors.shape[1], path=FAISS_PATH)
    faiss_index.add(np.array(ids), vectors)
    faiss_index.save()

    # Elastic index
    print("Indexing to Elastic Cloud...")
    docs = []
    for i, c in enumerate(chunks):
        docs.append({
            "id": str(ids[i]),
            "text": c["text"],
            "meta": c.get("meta", {})
        })

    elastic.bulk_index(docs)

    # Save docstore
    existing = load_docstore(DOCSTORE_PATH)
    for i, c in enumerate(chunks):
        existing[str(ids[i])] = {
            "meta": c.get("meta", {}),
            "text": c["text"]
        }

    save_docstore(DOCSTORE_PATH, existing)

    print("Ingest completed successfully.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-pdf>")
        sys.exit(1)

    main(sys.argv[1])
