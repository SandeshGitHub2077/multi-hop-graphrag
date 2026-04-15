# Chat-Style Document Upload RAG System

## Requirements
- ChatGPT/OpenUI-style interface with document upload/drag-and-drop
- Only uploaded documents are processed and used by RAG
- Users ask questions, system answers from their documents
- Interface similar to https://chat.pageindex.ai/

## Constraints
- Storage: Ephemeral (temp files), processed in-memory/temp storage
- Interface: HTML/JS web UI (embedded in FastAPI)
- Processing: Full pipeline (embed + graph + vector store)

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   HTML/JS UI    │────▶│   FastAPI API   │────▶│  RAG Pipeline   │
│ (drag & drop)   │     │ (upload endpoint)│    │ (parsing/graph) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Implementation Phases

### Phase 1: Backend Extensions
1. **Upload Endpoint** (`POST /upload`)
   - Accept multiple files (PDF, DOCX, TXT, MD)
   - Parse and process through full pipeline
   - Generate embeddings + store in Neo4j/FAISS
   - Return session/document IDs

2. **Session Management**
   - Track uploaded docs per session (in-memory or temp directory)
   - Isolate user queries to their uploaded documents

3. **Query Endpoint Update** (`POST /query`)
   - Accept session/doc filter
   - Only search within user's uploaded documents

### Phase 2: Frontend (Chat-Style UI)
1. **Layout** (like chat.pageindex.ai):
   - Left sidebar: Uploaded documents list
   - Main area: Chat messages
   - Drag-drop zone at bottom or separate tab

2. **Components**:
   - File uploader with progress indicator
   - Chat bubbles (user question, AI answer)
   - Source citations with section references

3. **Integration**:
   - Call `/upload` on file drop
   - Call `/query` for questions
   - Stream responses or show loading

### Phase 3: Testing & Refinement
- Upload multiple files, verify all indexed
- Query across documents
- Test edge cases (large files, unsupported formats)

## Key Files to Modify/Create
- `scripts/api.py`: Add upload endpoint + session management
- `scripts/static/index.html`: New chat UI
- `scripts/static/js/app.js`: Frontend logic
- `scripts/static/css/style.css`: Styling