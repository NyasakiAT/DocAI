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
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

QDRANT_DOCS_COLLECTION = "docs"
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gpt-oss:20b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

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
    cursor.execute("INSERT INTO chunks (chunk_id, content_hash) VALUES (?, ?)", (chunk_id, content_hash))
    docdb.commit()

def get_embeddings_function():
     embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
     return embeddings

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
            merged_docs[doc.metadata['source'] + f"_{doc.metadata['page']}"] = doc
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
        qclient = QdrantClient(url=QDRANT_URL)
        vectorstore = QdrantVectorStore(
            client=qclient,
            collection_name=QDRANT_DOCS_COLLECTION,
            embedding=get_embeddings_function(),
        )
        return vectorstore.add_documents(chunks)
    else:
        print("QDRANT_URL is not set. Skipping adding to Qdrant.")
        return 0
    
def query_rag(query: str):
    if QDRANT_URL:
        qclient = QdrantClient(url=QDRANT_URL)
        vectorstore = QdrantVectorStore(
            client=qclient,
            collection_name=QDRANT_DOCS_COLLECTION,
            embedding=get_embeddings_function(),
        )
        relevant_docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"If needed use the following context to answer the users message:\n{context}\n\nAnswer the following question: {query}"
        response = llm.invoke(prompt)
        return response.content + f"\n\n<small>(Sourced from {', '.join([doc.metadata['source'] for doc in relevant_docs])})</small>"
    else:
        return "QDRANT_URL is not set. Cannot perform RAG query."


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("chat.html")

@socketio.on('send')
def on_send(data):
    query   = (data.get('text') or '')
    emit('message', f"{markdown2.markdown(query)}", broadcast=True)
    answer = query_rag(query)
    emit('message', f"{markdown2.markdown(answer)}", broadcast=True)

if __name__ == "__main__":
    initialize_db()
    docs = load_documents()
    prepared_docs = prepare_documents(docs)
    chunks = create_chunks(prepared_docs)
    print(f"Prepared {len(chunks)} chunks for Qdrant")
    added_docs = add_to_qrant(chunks)
    print(f"Added {len(added_docs)} chunks to Qdrant collection '{QDRANT_DOCS_COLLECTION}'")
    socketio.run(app, debug=True)