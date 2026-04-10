from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Retrieve top_k chunks from store
        results = self._store.search(question, top_k=top_k)

        # Build context from retrieved chunks
        context = "\n\n".join([r["content"] for r in results])

        # Build RAG prompt
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""

        # Call the LLM function
        answer = self._llm_fn(prompt)

        return answer
