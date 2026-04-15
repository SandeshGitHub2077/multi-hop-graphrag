#!/usr/bin/env python3
"""FastAPI REST API for the RAG system with document upload support."""

import warnings
import logging
import io
import sys
import os
import tempfile
import hashlib
import uuid
import shutil

original_stdout = sys.stdout
original_stderr = sys.stderr

sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from parsing import DocumentParser, Section
from graph import Neo4jGraph
from embeddings import EmbeddingEngine, VectorStore
from retrieval import HybridRetriever, QueryRouter
from utils.config import config
from utils.health import HealthChecker
from utils.llm import LLMWrapper

sys.stdout = original_stdout
sys.stderr = original_stderr

app = FastAPI(title="GraphRAG API", version="1.0.0")

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "scripts" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SESSIONS: Dict[str, Dict] = {}
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

vector_store: Optional[VectorStore] = None
embedding_engine: Optional[EmbeddingEngine] = None
graph: Optional[Neo4jGraph] = None
retriever: Optional[HybridRetriever] = None
llm_wrapper: Optional[LLMWrapper] = None


def get_llm() -> LLMWrapper:
    global llm_wrapper
    if llm_wrapper is None:
        llm_wrapper = LLMWrapper()
        llm_wrapper.load()
    return llm_wrapper


def get_retriever() -> HybridRetriever:
    global vector_store, embedding_engine, graph, retriever
    
    if retriever is None:
        vector_store = VectorStore()
        vector_store.load(config.index_dir)
        
        embedding_engine = EmbeddingEngine(config.embedding_model)
        embedding_engine.load_model()
        
        graph = Neo4jGraph(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
        graph.connect()
        
        retriever = HybridRetriever(vector_store, graph, embedding_engine)
    
    return retriever


def create_session() -> str:
    session_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    SESSIONS[session_id] = {
        "dir": session_dir,
        "doc_ids": set(),
        "vector_store": None,
    }
    return session_id


def process_uploaded_file(session_id: str, file_path: Path, filename: str) -> dict:
    if session_id not in SESSIONS:
        raise ValueError(f"Invalid session: {session_id}")
    
    session = SESSIONS[session_id]
    doc_id = filename
    
    while doc_id in session["doc_ids"]:
        base, ext = os.path.splitext(doc_id)
        doc_id = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
    
    session["doc_ids"].add(doc_id)
    
    section_parser = DocumentParser()
    sections = section_parser.parse_file(str(file_path))
    
    for s in sections:
        s.doc_id = doc_id
    
    if not sections:
        return {"doc_id": doc_id, "sections": 0}
    
    session_graph = Neo4jGraph(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
    session_graph.connect()
    
    try:
        session_graph.upsert_document(doc_id)
        
        new_sections = []
        for section in sections:
            session_graph.upsert_section(
                section_id=section.section_id,
                content=section.content,
                doc_id=doc_id,
                metadata={"references": section.references}
            )
            session_graph.create_doc_relationship(section.section_id, doc_id)
            
            for ref in section.references:
                try:
                    session_graph.create_reference_relationship(section.section_id, ref)
                except:
                    pass
            new_sections.append(section)
        
        if new_sections:
            session_graph.upsert_document(doc_id)
    finally:
        session_graph.close()
    
    emb_engine = get_retriever().embedding_engine
    
    section_map = {s.section_id: s.content for s in sections}
    texts = []
    section_ids = []
    sdoc_ids = []
    for section in new_sections:
        augmented = emb_engine._augment_with_references(
            section.content, section.references, section_map
        )
        texts.append(augmented)
        section_ids.append(section.section_id)
        sdoc_ids.append(doc_id)
    
    if texts:
        embeddings = emb_engine.embed_batch(texts)
        
        contents = [s.content for s in new_sections]
        
        if session["vector_store"] is None:
            session["vector_store"] = VectorStore(dimension=embeddings.shape[1])
            session["vector_store"].build_index(
                embeddings, section_ids, sdoc_ids, contents
            )
        else:
            session["vector_store"].add_embeddings(
                embeddings, section_ids, sdoc_ids, contents
            )
    
    return {"doc_id": doc_id, "sections": len(sections)}


class QueryRequest(BaseModel):
    query: str
    k: int = 10
    multi_hop: bool = False
    session_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    use_llm: bool = True


class QueryResponse(BaseModel):
    answer: Optional[str] = None
    results: List[dict]


class HealthResponse(BaseModel):
    status: str
    details: dict


@app.get("/health")
async def health() -> HealthResponse:
    try:
        r = get_retriever()
        health_checker = HealthChecker(graph=graph, vector_store=vector_store)
        results = health_checker.check_all()
        
        all_ok = all(ok for ok, _ in results.values())
        
        return HealthResponse(
            status="healthy" if all_ok else "degraded",
            details={name: ok for name, (ok, msg) in results.items()}
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/documents")
async def list_documents(session_id: Optional[str] = None) -> dict:
    if session_id and session_id in SESSIONS:
        session = SESSIONS[session_id]
        return {"documents": [{"doc_id": doc_id, "title": doc_id} for doc_id in session["doc_ids"]]}
    
    global graph
    if graph is None:
        return {"documents": []}
    
    try:
        docs = graph.get_all_documents()
        return {"documents": [{"doc_id": d.doc_id, "title": d.title} for d in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session")
async def create_new_session() -> dict:
    session_id = create_session()
    return {"session_id": session_id}


@app.get("/session/{session_id}")
async def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = SESSIONS[session_id]
    return {
        "session_id": session_id,
        "documents": list(session["doc_ids"]),
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
) -> dict:
    if not session_id:
        session_id = create_session()
    
    if session_id not in SESSIONS:
        session_id = create_session()
    
    session = SESSIONS[session_id]
    file_path = session["dir"] / file.filename
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        result = process_uploaded_file(session_id, file_path, file.filename)
        
        return {
            "session_id": session_id,
            "doc_id": result["doc_id"],
            "sections": result["sections"],
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    try:
        r = get_retriever()
        
        if req.session_id and req.session_id in SESSIONS:
            session = SESSIONS[req.session_id]
            if session["vector_store"] is not None:
                session_vs = session["vector_store"]
                session_graph = Neo4jGraph(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
                session_graph.connect()
                
                try:
                    temp_retriever = HybridRetriever(
                        session_vs, session_graph, r.embedding_engine, use_cache=False
                    )
                    
                    if req.multi_hop:
                        from retrieval import MultiHopRetriever
                        mh_retriever = MultiHopRetriever(temp_retriever)
                        results = mh_retriever.retrieve_with_hops(req.query, target_hops=2)
                    else:
                        results = temp_retriever.retrieve(req.query, k=req.k)
                finally:
                    session_graph.close()
            else:
                results = []
        else:
            if req.multi_hop:
                from retrieval import MultiHopRetriever
                mh_retriever = MultiHopRetriever(r)
                results = mh_retriever.retrieve_with_hops(req.query, target_hops=2)
            else:
                results = r.retrieve(req.query, k=req.k)
        
        answer = None
        if req.use_llm and results:
            try:
                context = "\n\n".join([
                    f"[{res.section_id}]\n{res.content}"
                    for res in results[:5]
                ])
                llm = get_llm()
                answer = llm.generate_with_context(req.query, context)
            except Exception as e:
                pass
        
        return QueryResponse(
            answer=answer,
            results=[
                {
                    "section_id": res.section_id,
                    "doc_id": res.doc_id,
                    "content": res.content[:500] + "..." if len(res.content) > 500 else res.content,
                    "score": res.score,
                    "source": res.source,
                }
                for res in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest() -> dict:
    return {"status": "not implemented - use bash scripts/ingest.sh"}


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document RAG - Upload & Query</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a1a; color: #e0e0e0; height: 100vh; display: flex; }
            
            #sidebar { width: 280px; background: #252525; border-right: 1px solid #333; padding: 20px; display: flex; flex-direction: column; }
            #main { flex: 1; display: flex; flex-direction: column; }
            
            h2 { font-size: 18px; margin-bottom: 20px; color: #fff; }
            
            .upload-zone {
                border: 2px dashed #444; border-radius: 8px; padding: 30px 20px; text-align: center; cursor: pointer; transition: all 0.2s;
            }
            .upload-zone:hover { border-color: #666; background: #2a2a2a; }
            .upload-zone.dragover { border-color: #4a9eff; background: #2a3a4a; }
            
            #doc-list { margin-top: 20px; flex: 1; overflow-y: auto; }
            .doc-item { padding: 8px 12px; margin-bottom: 8px; background: #2d2d2d; border-radius: 6px; font-size: 14px; }
            
            #chat-container { flex: 1; overflow-y: auto; padding: 20px; }
            #messages { max-width: 800px; margin: 0 auto; }
            
            .message { margin-bottom: 20px; }
            .message.user { text-align: right; }
            .message .bubble { display: inline-block; padding: 12px 16px; border-radius: 12px; max-width: 80%; text-align: left; }
            .message.user .bubble { background: #4a9eff; color: #fff; }
            .message.assistant .bubble { background: #333; color: #e0e0e0; }
            
            .message.assistant .sources { margin-top: 12px; font-size: 12px; color: #888; }
            .message.assistant .sources .source { background: #2a2a2a; padding: 6px 10px; margin: 4px 4px; border-radius: 4px; display: inline-block; }
            
            #input-area { padding: 20px; border-top: 1px solid #333; }
            #input-container { max-width: 800px; margin: 0 auto; display: flex; gap: 10px; }
            #query-input { flex: 1; padding: 12px 16px; border: none; border-radius: 8px; background: #333; color: #fff; font-size: 16px; }
            #query-input:focus { outline: none; box-shadow: 0 0 0 2px #4a9eff; }
            #send-btn { padding: 12px 24px; border: none; border-radius: 8px; background: #4a9eff; color: #fff; cursor: pointer; font-size: 16px; }
            #send-btn:hover { background: #3a8eef; }
            #send-btn:disabled { background: #444; cursor: not-allowed; }
            
            .loading { text-align: center; padding: 20px; color: #888; }
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h2>Documents</h2>
            <div class="upload-zone" id="upload-zone">
                Drop files here<br>or click to upload
                <input type="file" id="file-input" multiple style="display: none;">
            </div>
            <div id="doc-list"></div>
        </div>
        <div id="main">
            <div id="chat-container">
                <div id="messages"></div>
            </div>
            <div id="input-area">
                <div id="input-container">
                    <input type="text" id="query-input" placeholder="Ask a question about your documents...">
                    <button id="send-btn">Send</button>
                </div>
            </div>
        </div>
        <script>
            let sessionId = null;
            const messages = document.getElementById('messages');
            const queryInput = document.getElementById('query-input');
            const sendBtn = document.getElementById('send-btn');
            const uploadZone = document.getElementById('upload-zone');
            const fileInput = document.getElementById('file-input');
            const docList = document.getElementById('doc-list');
            
            async function createSession() {
                const res = await fetch('/session', { method: 'POST' });
                const data = await res.json();
                sessionId = data.session_id;
            }
            
            async function uploadFile(file) {
                const formData = new FormData();
                formData.append('file', file);
                if (sessionId) formData.append('session_id', sessionId);
                
                const res = await fetch('/upload', { method: 'POST', body: formData });
                if (!res.ok) throw new Error('Upload failed');
                return res.json();
            }
            
            async function sendQuery(text) {
                const res = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: text, session_id: sessionId, k: 10 })
                });
                if (!res.ok) throw new Error('Query failed');
                return res.json();
            }
            
            function addMessage(role, content, sources = []) {
                const div = document.createElement('div');
                div.className = `message ${role}`;
                div.innerHTML = `<div class="bubble">${content}</div>`;
                
                if (sources.length > 0 && role === 'assistant') {
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.className = 'sources';
                    sourcesDiv.innerHTML = sources.map(s => 
                        `<div class="source">${s.doc_id} - ${s.section_id}</div>`
                    ).join('');
                    div.appendChild(sourcesDiv);
                }
                
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }
            
            async function loadDocuments() {
                if (!sessionId) return;
                const res = await fetch(`/session/${sessionId}`);
                const data = await res.json();
                docList.innerHTML = data.documents.map(doc => 
                    `<div class="doc-item">${doc}</div>`
                ).join('');
            }
            
            uploadZone.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', async (e) => {
                for (const file of e.target.files) {
                    try {
                        const result = await uploadFile(file);
                        addMessage('assistant', `Uploaded: ${file.name} (${result.sections} sections)`);
                        await loadDocuments();
                    } catch (err) {
                        addMessage('assistant', `Error: ${err.message}`);
                    }
                }
            });
            
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });
            uploadZone.addEventListener('drop', async (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                for (const file of e.dataTransfer.files) {
                    try {
                        const result = await uploadFile(file);
                        addMessage('assistant', `Uploaded: ${file.name} (${result.sections} sections)`);
                        await loadDocuments();
                    } catch (err) {
                        addMessage('assistant', `Error: ${err.message}`);
                    }
                }
            });
            
            sendBtn.addEventListener('click', async () => {
                const text = queryInput.value.trim();
                if (!text) return;
                queryInput.value = '';
                addMessage('user', text);
                sendBtn.disabled = true;
                
                try {
                    const data = await sendQuery(text);
                    const answer = data.answer || (data.results.length > 0 
                        ? data.results.map(r => r.content).join('\\n\\n')
                        : 'No results found.');
                    addMessage('assistant', answer, data.results);
                } catch (err) {
                    addMessage('assistant', `Error: ${err.message}`);
                }
                sendBtn.disabled = false;
            });
            
            queryInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendBtn.click();
            });
            
            createSession().then(loadDocuments);
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)