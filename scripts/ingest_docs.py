#!/usr/bin/env python3

import json
from pathlib import Path

from app.rag.ingestor import ingest_documentation

def main():
    """Ingest documentation into Qdrant vector database."""
    try:
        docs_dir = Path(__file__).resolve().parent.parent / "docs"
        
        if not docs_dir.exists():
            print(f"❌ Docs directory not found: {docs_dir}")
            print(f"   Please ensure documentation files exist in: {docs_dir}")
            return 1
            
        print(f"🚀 Starting documentation ingestion from: {docs_dir}")
        print(f"📄 Documents available: {list(docs_dir.glob('*'))}")
        
        chunks_stored = ingest_documentation(docs_dir)
        
        print(f"✅ Successfully ingested {chunks_stored} document chunks")
        print(f"📊 Chunks stored in Qdrant collection: docs")
        print(f"🎯 RAG pipeline is now ready for queries")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please check that all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
