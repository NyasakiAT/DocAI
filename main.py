import os
import hashlib
import uuid
import sqlite3
import markdown2
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

QDRANT_DOCS_COLLECTION = "docs"
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gpt-oss:20b")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
#DENSE_EMBED_MODEL = os.getenv("DENSE_EMBED_MODEL", "nomic-ai/nomic-embed-text-v2-moe")
dense_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

sparse_model = SparseTextEmbedding("Qdrant/bm25")
llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0.2)
docdb = sqlite3.connect("document-memory.db")
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def initialize_db():
    cursor = docdb.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content_hash TEXT KEY
                )
            ''')
    docdb.commit()

def check_chunk_changed(chunk_id, content_hash):
    cursor = docdb.cursor()
    print(f"Checking chunk {chunk_id}")
    cursor.execute("SELECT content_hash FROM chunks WHERE chunk_id = ?", (chunk_id,))
    result = cursor.fetchone()
    if result:
        return result[0] != content_hash
    else:
        return True  # Chunk does not exist, so it is considered changed

def add_chunk(chunk_id, content_hash):
    cursor = docdb.cursor()
    cursor.execute("""
        INSERT INTO chunks (chunk_id, content_hash) VALUES (?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET content_hash=excluded.content_hash
    """, (chunk_id, content_hash))
    docdb.commit()

def load_documents():
    loaders = [
        DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader("data", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, loader_kwargs={"mode": "elements"}),
        DirectoryLoader("data", glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader, loader_kwargs={"mode": "elements"}),
        PyPDFDirectoryLoader("data")
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents

def prepare_documents(documents):
    # Word and markdown documents have to be merged because they are split into multiple elements
    merged_docs = {}

    for doc in documents:
        if doc.metadata['source'].endswith('.docx') or doc.metadata['source'].endswith('.md'):
            if doc.metadata['source'] not in merged_docs:
                merged_docs[doc.metadata['source']] = doc
            else:
                merged_docs[doc.metadata['source']].page_content += "\n" + doc.page_content
        else:
            # For PDFs and other files, keep them as is
            merged_docs[doc.metadata['source'] + f"_{doc.metadata['page'] if 'page' in doc.metadata else '0'}"] = doc
    return list(merged_docs.values())

def create_chunks(documents):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            hex_id = hashlib.sha256(f"{doc.metadata['source']}:{doc.metadata['page'] if 'page' in doc.metadata else '0'}:{i}".encode()).hexdigest()

            chunk_id = str(uuid.UUID(hex_id[:32]))
            if check_chunk_changed(chunk_id, hashlib.sha256(chunk.encode()).hexdigest()):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata['chunk_no'] = i
                all_chunks.append(Document(id=chunk_id, vector=None, page_content=chunk, metadata=chunk_metadata))
                add_chunk(chunk_id, hashlib.sha256(chunk.encode()).hexdigest())
            else:
                pass
                #print(f"Chunk {i} of document {doc.metadata['source']} unchanged, skipping.")
         
    return all_chunks

def add_to_qrant(chunks):
    if QDRANT_URL:
        qclient = QdrantClient(url=QDRANT_URL)       # your llama.cpp/nomic embedder
        dense_vecs = dense_model.encode([c.page_content for c in chunks], normalize_embeddings=True)
        sparse_vecs = list(sparse_model.embed([c.page_content for c in chunks]))
        points = []
        for i, c in enumerate(chunks):
            # Convert sparse vector to Qdrant SparseVector
            sv = sparse_vecs[i]
            # FastEmbed returns something with .indices and .values
            sparse_vector = qm.SparseVector(
                indices=list(map(int, sv.indices)),
                values=list(map(float, sv.values))
            )

            points.append(
                qm.PointStruct(
                    id=i,
                    vector={
                        "dense": dense_vecs[i].tolist(),
                        "sparse": sparse_vector, 
                    },
                    payload={
                        "text": c.page_content,
                        "metadata": c.metadata,
                    },
                )
            )
        qclient.upsert("docs", points)
        return len(points)

def _to_qdrant_sparse(fe_sparse):
    # FastEmbed `embed()` element -> qm.SparseVector
    idxs = list(map(int, fe_sparse.indices))
    vals = list(map(float, fe_sparse.values))
    return qm.SparseVector(indices=idxs, values=vals)
    
def query_rag(query: str, k: int = 3, alpha: float = 0.7):
    print(f"RAG Query: {query}")
    if QDRANT_URL:
        qclient = QdrantClient(url=QDRANT_URL)

        dense_q = dense_model.encode([query], normalize_embeddings=True)[0]

        sparse_q_fe = list(sparse_model.embed([query]))[0]
        sparse_q = _to_qdrant_sparse(sparse_q_fe)

        res = qclient.query_points(
            collection_name=QDRANT_DOCS_COLLECTION,
            query=dense_q.tolist(),         
            using="dense", 
            prefetch=[
                qm.Prefetch(query=dense_q, using="dense", limit=max(20, k*4)),
                qm.Prefetch(query=sparse_q, using="sparse", limit=max(20, k*4)),
            ],
            limit=max(20, k*4),          
            with_payload=True,
            with_vectors=False,
            search_params=qm.SearchParams(hnsw_ef=128, exact=False),
        )

        # 3) Take top-k (Qdrant blends scores internally in prefetch mode)
        points = res.points[:k]

        relevant_docs = []
        for p in points:
            text = p.payload.get("text", "")
            metadata = p.payload.copy()
            metadata.pop("text", None)
            relevant_docs.append(Document(page_content=text, metadata=metadata))

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"If needed use the following context to answer the users message:\n{context}\n\nAnswer the following question: {query}"

        response = llm.invoke(prompt)
        sources = ", ".join(doc.metadata['metadata']['source'] for doc in relevant_docs)
        print("Query Done.")
        return f"{response.content}\n\n<small>(Sourced from {sources})</small>"
    else:
        return "QDRANT_URL is not set. Cannot perform RAG query."

def _process_and_reply(sid, query):
    try:
        socketio.emit('loading', True, to=sid)
        socketio.sleep(0)
        answer = query_rag(query)
        socketio.emit('message', markdown2.markdown(answer), to=sid)
    except Exception as e:
        socketio.emit('message', f"<pre>{e}</pre>", to=sid)
    finally:
        socketio.emit('loading', False, to=sid)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("chat.html")

@socketio.on('send')
def on_send(data):
    sid = request.sid
    query = (data.get('text') or '')
    emit('message', markdown2.markdown(query), to=sid)
    socketio.start_background_task(_process_and_reply, sid, query)

if __name__ == "__main__":
    initialize_db()
    docs = load_documents()
    prepared_docs = prepare_documents(docs)
    chunks = create_chunks(prepared_docs)
    print(f"Prepared {len(chunks)} chunks for Qdrant")
    if len(chunks) > 0:
        added_docs = add_to_qrant(chunks)
        print(f"Added {added_docs} chunks to Qdrant collection '{QDRANT_DOCS_COLLECTION}'")
    socketio.run(app)