from langchain_core.documents import Document
from langchain_ollama import ChatOllama

class Rag:
    OLLAMA_CHAT_MODEL = None
    QDRANT_HANDLER = None

    def __init__(self, model:str, qdrant_handler):
        self.OLLAMA_CHAT_MODEL = model
        self.QDRANT_HANDLER = qdrant_handler

    def query_rag(self, query: str):
        llm = ChatOllama(model=self.OLLAMA_CHAT_MODEL, temperature=0.2)
        print(f"RAG Query: {query}")
        
        points = self.QDRANT_HANDLER.query_qdrant(query)

        relevant_docs = []
        for p in points:
            text = p.payload.get("text", "")
            metadata = p.payload.copy()
            metadata.pop("text", None)
            relevant_docs.append(Document(page_content=text, metadata=metadata))

        context = "\n\n".join([f"{doc.metadata['source']}:\n{doc.page_content}" for doc in relevant_docs])
        prompt = f"If needed use the following context to answer the users message:\n{context}\n\nAnswer the following question: {query}"

        response = llm.invoke(prompt)
        sources = ", ".join(doc.metadata['source'] for doc in relevant_docs)
        print("Query Done.")
        return f"{response.content}\n\n<small>(Sourced from {sources})</small>"
