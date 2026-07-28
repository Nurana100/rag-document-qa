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
    return vector_store.similarity_search(query, k=k)


def build_prompt(query, retrieved_docs):
    """
    Builds a prompt with clear separation between retrieved context
    and the actual question. Each chunk is labeled [Source N] so the
    model (and we, afterward) can reference exactly which chunk was used.
    """
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        context_blocks.append(f"[Source {i}]\n{doc.page_content}")

    context_text = "\n\n---\n\n".join(context_blocks)

    user_message = f"""CONTEXT:
{context_text}

---

QUESTION:
{query}

Answer the question using only the CONTEXT above."""

    return user_message


def format_sources(retrieved_docs):
    """
    Builds a human-readable list of where each retrieved chunk came from:
    file name + a short preview of the chunk content.
    """
    lines = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source_file = doc.metadata.get("source", "unknown file")
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:80] + "..."
        lines.append(f"  [Source {i}] {source_file} — \"{preview}\"")
    return "\n".join(lines)


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
        "What happens if I submit an expense report after 30 days?",
        "What is the company's policy on annual bonuses?"  # NOT in the handbook — tests hallucination handling
    ]

    for question in test_questions:
        print(f"Question: {question}")
        answer, sources = ask(question, store)
        print(f"Answer: {answer}\n")
        print("Sources used:")
        print(format_sources(sources))
        print("=" * 60)