from config import settings


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build a RAG prompt by injecting retrieved chunks as numbered context blocks.

    Format:
        [Context 1] (source.pdf, page 3)
        <chunk text>

        [Context 2] ...

        Question: <question>
        Answer:
    """
    if not chunks:
        return (
            f"No relevant context was found in the knowledge base.\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Context {i}] ({chunk['source']}, page {chunk['page']})"
        context_blocks.append(f"{header}\n{chunk['text']}")

    context_str = "\n\n".join(context_blocks)

    prompt = (
        f"{context_str}\n\n"
        f"---\n"
        f"Using only the context above, answer the following question.\n"
        f"If the context doesn't contain the answer, say \"I don't have enough "
        f"information in the provided documents to answer this.\"\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    return prompt


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """
    Return Ollama-compatible messages array (system + user turn).
    Use this when calling the /api/chat endpoint instead of /api/generate.
    """
    return [
        {"role": "system", "content": settings.system_prompt},
        {"role": "user", "content": build_prompt(question, chunks)},
    ]