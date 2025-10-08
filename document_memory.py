import sqlite3
import hashlib
import uuid
from langchain_core.documents import Document

class DocumentMemory:
    DOC_DB = None
    CHUNK_SIZE = None
    CHUNK_OVERLAP = None

    def __init__(self, db_name:str, chunk_size:int, chunk_overlap:int):
        self.CHUNK_SIZE = chunk_size
        self.CHUNK_OVERLAP = chunk_overlap
        self.DOC_DB = sqlite3.connect(db_name)

    def initialize_db(self):
        cursor = self.DOC_DB.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        content_hash TEXT KEY
                    )
                ''')
        self.DOC_DB.commit()

    def check_chunk_changed(self, chunk_id, content_hash):
        cursor = self.DOC_DB.cursor()
        #print(f"Checking chunk {chunk_id}")
        cursor.execute("SELECT content_hash FROM chunks WHERE chunk_id = ?", (chunk_id,))
        result = cursor.fetchone()
        if result:
            return result[0] != content_hash
        else:
            return True  # Chunk does not exist, so it is considered changed

    def add_chunk(self, chunk_id, content_hash):
        cursor = self.DOC_DB.cursor()
        cursor.execute("""
            INSERT INTO chunks (chunk_id, content_hash) VALUES (?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET content_hash=excluded.content_hash
        """, (chunk_id, content_hash))
        self.DOC_DB.commit()

    def create_chunks(self, documents):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP)
        all_chunks = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                hex_id = hashlib.sha256(
                    f"{doc.metadata['source']}:{doc.metadata['page'] if 'page' in doc.metadata else '0'}:{i}".encode()).hexdigest()

                chunk_id = str(uuid.UUID(hex_id[:32]))
                if self.check_chunk_changed(chunk_id, hashlib.sha256(chunk.encode()).hexdigest()):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata['chunk_no'] = i
                    all_chunks.append(Document(id=chunk_id, vector=None, page_content=chunk, metadata=chunk_metadata))
                    self.add_chunk(chunk_id, hashlib.sha256(chunk.encode()).hexdigest())
                else:
                    pass
                    # print(f"Chunk {i} of document {doc.metadata['source']} unchanged, skipping.")

        return all_chunks