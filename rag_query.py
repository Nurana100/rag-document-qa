import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from vectorstore import load_vector_store

load_dotenv()

client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
context provided below. Do not use any outside knowledge.

Rules:
- Base your answer strictly on the CONTEXT section.
- If the answer is not contained in the CONTEXT, say clearly that you don't know
  based on the available documents. Do not guess or make up an answer.
- Keep answers concise and directly address the question."""


def retrieve_context(query, vector_store, k=2):
    """
    Retrieves the top-k most relevant chunks for the query.
    Returns both the combined text (for the prompt) and the raw
    documents (so we can cite sources later in Checkpoint 5).
    """
    results = vector_store.similarity_search(query, k=k)
    return results


def build_prompt(query, retrieved_docs):
    """
    Builds a prompt with clear separation between retrieved context
    and the actual question/instructions.
    """
    context_text = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    user_message = f"""CONTEXT:
{context_text}

---

QUESTION:
{query}

Answer the question using only the CONTEXT above."""

    return user_message


def ask(query, vector_store, k=2):
    retrieved_docs = retrieve_context(query, vector_store, k=k)
    user_message = build_prompt(query, retrieved_docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        max_tokens=200
    )

    answer = response.choices[0].message.content
    return answer, retrieved_docs


if __name__ == "__main__":
    store = load_vector_store()

    test_questions = [
        "How many vacation days do employees get?",
        "What happens if I submit an expense report after 30 days?"
    ]

    for question in test_questions:
        print(f"Question: {question}")
        answer, sources = ask(question, store)
        print(f"Answer: {answer}\n")
        print(f"Retrieved from {len(sources)} chunk(s).")
        print("=" * 60)