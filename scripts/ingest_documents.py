import warnings
import logging
import sys
import io
import os

original_stdout = sys.stdout
original_stderr = sys.stderr

sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

import argparse
import hashlib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from parsing import DocumentParser
from graph import Neo4jGraph
from embeddings import EmbeddingEngine, VectorStore

sys.stdout = original_stdout
sys.stderr = original_stderr


def get_file_hash(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("--data-dir", default="Data", help="Directory containing documents")
    parser.add_argument("--graph-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--graph-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--graph-password", default="password", help="Neo4j password")
    parser.add_argument("--index-dir", default="index", help="Directory to save FAISS index")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5", help="Embedding model")
    parser.add_argument("--rebuild", action="store_true", help="Full rebuild (delete existing data first)")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Parsing Documents")
    print("=" * 60)
    
    parser_obj = DocumentParser(args.data_dir)
    sections = parser_obj.parse_all()
    
    print(f"\nTotal sections extracted: {len(sections)}")

    print("\n" + "=" * 60)
    print("Step 2: Building Knowledge Graph (Neo4j)")
    print("=" * 60)
    
    graph = Neo4jGraph(args.graph_uri, args.graph_user, args.graph_password)
    try:
        graph.connect()
        graph.ensure_constraints()
        
        if args.rebuild:
            graph.clear_all()
            print("Full rebuild mode - cleared existing data")
        
        existing_sections = graph.get_section_ids()
        existing_docs = set(graph.get_all_documents())
        print(f"Existing sections in graph: {len(existing_sections)}")
        print(f"Existing documents in graph: {len(existing_docs)}")
        
        new_sections = [s for s in sections if s.section_id not in existing_sections]
        print(f"New sections to add: {len(new_sections)}")
        
        unique_docs = set(s.doc_id for s in sections)
        for doc_id in unique_docs:
            if doc_id not in existing_docs:
                graph.upsert_document(doc_id)
                print(f"Created document node: {doc_id}")
        
        for section in new_sections:
            graph.upsert_section(
                section_id=section.section_id,
                content=section.content,
                doc_id=section.doc_id,
                metadata={"references": section.references}
            )
            graph.create_doc_relationship(section.section_id, section.doc_id)
            
            for ref in section.references:
                try:
                    graph.create_reference_relationship(
                        section.section_id, ref
                    )
                except Exception as e:
                    print(f"Warning: Could not create reference {section.section_id} -> {ref}: {e}")
        
        if new_sections:
            print(f"\nAdded {len(new_sections)} new sections to graph")
        else:
            print("\nNo new sections to add")
    finally:
        graph.close()

    print("\n" + "=" * 60)
    print("Step 3: Building Embeddings and Vector Index")
    print("=" * 60)
    
    index_dir = Path(args.index_dir)
    
    embedding_engine = EmbeddingEngine(args.embedding_model)
    embedding_engine.load_model()
    
    new_section_ids = [s.section_id for s in new_sections]
    new_doc_ids = [s.doc_id for s in new_sections]
    section_map = {s.section_id: s.content for s in sections}
    
    texts = []
    section_ids = []
    doc_ids = []
    for section in new_sections:
        augmented = embedding_engine._augment_with_references(
            section.content, section.references, section_map
        )
        texts.append(augmented)
        section_ids.append(section.section_id)
        doc_ids.append(section.doc_id)
    
    if texts:
        batch_size = 100
        all_embeddings = []
        
        print(f"Generating embeddings for {len(texts)} sections (batch size: {batch_size})...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = embedding_engine.embed_batch(batch)
            all_embeddings.append(batch_embeddings)
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} sections")
        
        new_embeddings = np.vstack(all_embeddings)
        print(f"Generated embeddings for {len(texts)} new sections")
        
        if not args.rebuild and index_dir.exists():
            vector_store = VectorStore()
            vector_store.load(str(index_dir))
            
            new_contents = [s.content for s in new_sections]
            vector_store.add_embeddings(new_embeddings, section_ids, doc_ids, new_contents)
        else:
            new_contents = [s.content for s in new_sections]
            vector_store = VectorStore(dimension=new_embeddings.shape[1])
            vector_store.build_index(new_embeddings, section_ids, doc_ids, new_contents)
        
        index_dir.mkdir(exist_ok=True)
        vector_store.save(str(index_dir))
        print(f"Vector index saved to {index_dir}")
    else:
        print("No new sections to embed")
    
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
