from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import markdown
from document_memory import DocumentMemory
from document_handler import DocumentHandler
from qdrant_handler import QdrantHandler
from rag import Rag

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", logger=True)#
docs_db = DocumentMemory("docs.sqlite", 500, 100)
docs_handler = DocumentHandler("data")
qdrant_handler = QdrantHandler("http://localhost:6333", "docs")
rag = Rag("gpt-oss:20b", qdrant_handler)

def _process_and_reply(self, query):
    try:
        socketio.emit('loading', True)
        socketio.sleep(0)
        answer = rag.query_rag(query)
        socketio.emit('message', markdown.markdown(answer, extensions=["tables", "fenced_code"]))
    except Exception as e:
        socketio.emit('message', f"<pre>{e}</pre>")
    finally:
        socketio.emit('loading', False)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("chat.html")

@socketio.on('send')
def on_send(data):
    query = (data.get('text') or '')
    emit('message', markdown.markdown(query))
    socketio.start_background_task(_process_and_reply, request.sid, query)

if __name__ == "__main__":
    docs_db.initialize_db()
    docs = docs_handler.load_documents()
    prepared_docs = docs_handler.prepare_documents(docs)
    chunks = docs_db.create_chunks(prepared_docs)
    print(f"Prepared {len(chunks)} chunks for Qdrant")
    added_docs = qdrant_handler.add_to_qrant(chunks)
    socketio.run(app)
