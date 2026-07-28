\# RAG Document Q\&A



A Retrieval-Augmented Generation (RAG) pipeline that answers questions about a document using semantic search and an LLM — instead of relying on the model's general knowledge, answers are grounded in the actual source document.



\## How it works



1\. \*\*Ingestion + chunking\*\* (`ingest.py`) — loads the document and splits it into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`, tuned to keep related sentences from being split across chunk boundaries.

2\. \*\*Embeddings\*\* (`embed.py`) — converts each chunk into a 384-dimensional vector using a free, local model (`sentence-transformers/all-MiniLM-L6-v2`), so no API key or cost is needed for this step.

3\. \*\*Vector storage + similarity search\*\* (`vectorstore.py`) — stores embeddings in a local Chroma database (persisted to `chroma\_db/`) and retrieves the most relevant chunks for a given question.

4\. \*\*Retrieval + prompt construction\*\* (`rag\_query.py`) — builds a prompt with a clearly separated `CONTEXT` block (retrieved chunks) and `QUESTION` block, so the model doesn't confuse the two.

5\. \*\*Source tracking\*\* — each answer is accompanied by the source file and a preview of the chunk(s) it was generated from.

6\. \*\*Hallucination handling\*\* — if a question isn't answerable from the document, the model is instructed to say so rather than guess.



\## Setup



1\. Clone this repo and navigate into it:

```bash

&#x20;  git clone https://github.com/Nurana100/rag-document-qa.git

&#x20;  cd rag-document-qa

```



2\. Create and activate a virtual environment:

```bash

&#x20;  python -m venv venv

&#x20;  venv\\Scripts\\activate   # Windows

&#x20;  source venv/bin/activate   # Mac/Linux

```



3\. Install dependencies:

```bash

&#x20;  pip install langchain langchain-community langchain-huggingface langchain-chroma chromadb python-dotenv huggingface\_hub pypdf sentence-transformers

```



4\. Copy `.env.example` to `.env` and add your Hugging Face API key.

Get a free token at \[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — when creating it, use the "Inference" preset so it has permission to call Inference Providers.



\## Usage



Build the vector store and run a query:

```bash

python rag\_query.py

```



This runs a set of test questions against `company\_handbook.txt` and prints each answer along with its source chunk(s).



\## Sample request/response log



\*\*Question:\*\* How many vacation days do employees get?

\*\*Answer:\*\* Employees get 20 days of paid vacation per year.

\*\*Sources used:\*\* `company\_handbook.txt` — Section 1: Vacation Policy



\---



\*\*Question:\*\* What happens if I submit an expense report after 30 days?

\*\*Answer:\*\* If you submit an expense report after 30 days, it requires additional manager approval and may be denied at the finance team's discretion.

\*\*Sources used:\*\* `company\_handbook.txt` — Section 3: Expense Reimbursement



\---



\*\*Question:\*\* What is the company's policy on annual bonuses?

\*\*Answer:\*\* I don't know based on the available documents. The provided context only discusses vacation policy and expense reimbursement, but does not mention annual bonuses.

\*\*Sources used:\*\* `company\_handbook.txt` — Section 1 and Section 3 (retrieved, but do not contain a bonus policy)



\## Tech stack



\- Python

\- LangChain (document loading, chunking, retrieval)

\- Hugging Face Inference API (LLM: `meta-llama/Llama-3.1-8B-Instruct`)

\- sentence-transformers (local embeddings: `all-MiniLM-L6-v2`)

\- Chroma (vector database)

