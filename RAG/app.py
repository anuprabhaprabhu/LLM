from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


if __name__ == "__main__":
    
    # 1. ALWAYS load documents
    docs = load_all_documents(DATA_PATH)

    # 2. Initialize vector store
    store = FaissVectorStore(FAISS_PATH)

    # 3. Build or load index
    if faiss_index_exists(FAISS_PATH):
        print("Loading existing FAISS index...")
        store.load()
    else:
        print("FAISS index not found. Building from documents...")
        store.build_from_documents(docs)
        store.save()

    # 4. RAG search
    rag_search = RAGSearch(vector_store=store)

    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)

    print("\nSummary:")
    print(summary)
