from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_and_chunk

def get_embedding_model():
    """
    Loads a free, local sentence-embedding model.
    all-MiniLM-L6-v2 is small, fast, and runs on CPU with no API key needed.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if __name__ == "__main__":
    chunks = load_and_chunk("company_handbook.txt")
    embedding_model = get_embedding_model()

    texts = [chunk.page_content for chunk in chunks]
    vectors = embedding_model.embed_documents(texts)

    print(f"Generated {len(vectors)} embeddings.")
    print(f"Each embedding has {len(vectors[0])} dimensions.")
    print(f"\nFirst 10 values of Chunk 1's embedding vector:")
    print(vectors[0][:10])