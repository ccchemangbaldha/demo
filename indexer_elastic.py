# indexer_elastic.py
from elasticsearch import Elasticsearch, helpers

class ElasticIndexer:
    def __init__(self, cloud_id: str, api_key: str, index: str):
        if not cloud_id or not api_key:
            raise ValueError("Elastic Cloud ID and API Key required")

        self.client = Elasticsearch(
            cloud_id=cloud_id,
            api_key=api_key,
        )

        self.index = index

        # Create index with mapping if not exists
        if not self.client.indices.exists(index=self.index):
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "subject": {"type": "keyword"},
                        "class": {"type": "integer"},
                        "chapter": {"type": "integer"},
                        "section": {"type": "keyword"},
                        "type": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "chunk_start_word": {"type": "integer"}
                    }
                }
            }
            self.client.indices.create(index=self.index, body=mapping)

    def bulk_index(self, docs):
        actions = []
        for d in docs:
            actions.append({
                "_index": self.index,
                "_id": d["id"],
                "_source": {
                    "text": d["text"],
                    **d.get("meta", {})
                }
            })

        helpers.bulk(self.client, actions, refresh=True)

    def search_bm25(self, query, size=20, filters=None):
        must = [{"match": {"text": {"query": query}}}]

        if filters:
            for k, v in filters.items():
                must.append({"term": {k: v}})

        body = {"query": {"bool": {"must": must}}}

        res = self.client.search(index=self.index, body=body, size=size)
        hits = res["hits"]["hits"]

        return [(h["_id"], h["_score"], h["_source"]) for h in hits]

    def get_docs(self, ids):
        res = self.client.mget(index=self.index, body={"ids": ids})
        return [d for d in res["docs"] if d.get("found")]
