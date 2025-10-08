from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding


class QdrantHandler:

    Q_CLIENT = None
    QDRANT_URL = None
    QDRANT_COLLECTION = None
    K = 10
    ALPHA = 0.7

    DENSE_MODEL = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    SPARSE_MODEL = SparseTextEmbedding("Qdrant/bm25")

    def __init__(self, qdrant_url:str, qdrant_collection:str):
        self.QDRANT_URL = qdrant_url
        self.QDRANT_COLLECTION = qdrant_collection
        self.Q_CLIENT = QdrantClient(url=self.QDRANT_URL, prefer_grpc=True)

    def upsert_in_batches(self, client, collection, points, max_points=100, max_bytes=8_000_000):
        batch, byte_budget = [], 0
        for p in points:
            t = (p.payload or {}).get("text", "")
            size = len(t.encode("utf-8")) + 1024
            if batch and (len(batch) >= max_points or byte_budget + size > max_bytes):
                client.upsert(collection_name=collection, points=batch)
                batch, byte_budget = [], 0
            batch.append(p)
            byte_budget += size
        if batch:
            client.upsert(collection_name=collection, points=batch)

    def add_to_qrant(self, chunks):
        if self.QDRANT_URL and len(chunks) > 0:
            dense_vecs = self.DENSE_MODEL.encode([c.page_content for c in chunks], normalize_embeddings=True)
            sparse_vecs = list(self.SPARSE_MODEL.embed([c.page_content for c in chunks]))
            points = []
            for i, c in enumerate(chunks):
                sv = sparse_vecs[i]
                sparse_vector = qm.SparseVector(
                    indices=list(map(int, sv.indices)),
                    values=list(map(float, sv.values))
                )

                points.append(
                    qm.PointStruct(
                        id=c.id,
                        vector={
                            "dense": dense_vecs[i].tolist(),
                            "sparse": sparse_vector, 
                        },
                        payload={
                            "text": c.page_content,
                            "source": c.metadata.get("source"),
                            "page": c.metadata.get("page"),
                            "chunk_no": c.metadata.get("chunk_no"),
                            "chunk_id": c.id,
                        },
                    )
                )
            self.upsert_in_batches("docs", points, max_points=100, max_bytes=8_000_000)
            self.Q_CLIENT.upsert("docs", points)
            print(f"Added {len(chunks)} chunks to Qdrant collection '{self.QDRANT_COLLECTION}'")
            return len(points)

    def _to_qdrant_sparse(self, fe_sparse):
        idxs = list(map(int, fe_sparse.indices))
        vals = list(map(float, fe_sparse.values))
        return qm.SparseVector(indices=idxs, values=vals)
    
    def query_qdrant(self, query:str):
        if self.QDRANT_URL:
            dense_q = self.DENSE_MODEL.encode([query], normalize_embeddings=True)[0]

            sparse_q_fe = list(self.SPARSE_MODEL.embed([query]))[0]
            sparse_q = self._to_qdrant_sparse(sparse_q_fe)

            res = self.Q_CLIENT.query_points(
                collection_name=self.QDRANT_COLLECTION,
                query=dense_q.tolist(),         
                using="dense", 
                prefetch=[
                    qm.Prefetch(query=dense_q, using="dense", limit=max(20, self.K*4)),
                    qm.Prefetch(query=sparse_q, using="sparse", limit=max(20, self.K*4)),
                ],
                limit=max(20, self.K*4),          
                with_payload=True,
                with_vectors=False,
                search_params=qm.SearchParams(hnsw_ef=128, exact=False),
            )

            return res.points[:self.K]