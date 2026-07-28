from langchain_chroma import Chroma
from ingest import load_and_chunk
from embed import get_embedding_model

PERSIST_DIR = "chroma_db"

def build_vector_store(filepath="company_handbook.txt"):
    """
    Loads, chunks, embeds, and stores the document in a local Chroma
    vector database, persisted to disk so it doesn't need rebuilding
    every time.
    """
    chunks = load_and_chunk(filepath)
    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR
    )
    return vector_store

def load_vector_store():
    """Loads an existing Chroma store from disk without rebuilding it."""
    embedding_model = get_embedding_model()
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )

if __name__ == "__main__":
    print("Building vector store...")
    store = build_vector_store()
    print("Vector store built and saved to disk.\n")

    # Test similarity search with a question that doesn't use the document's exact wording
    test_queries = [
        "How much time off do I get per year?",
        "Can I work from home?",
        "What happens if I submit an expense report late?"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        results = store.similarity_search(query, k=1)
        print(f"Best match:\n{results[0].page_content}\n")
        print("-" * 60)