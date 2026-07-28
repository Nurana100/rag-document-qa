from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(filepath, chunk_size=400, chunk_overlap=100):
    """
    Loads a document and splits it into overlapping chunks.
    - chunk_size: target size of each chunk (in characters)
    - chunk_overlap: how many characters overlap between consecutive chunks,
      so that information near a chunk boundary isn't lost entirely from either side.
    """
    loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # try to split on paragraph/sentence boundaries first
    )

    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk("company_handbook.txt")
    print(f"Total chunks created: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ({len(chunk.page_content)} chars) ---")
        print(chunk.page_content)
        print()