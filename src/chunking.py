from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Split on ". ", "! ", "? " or ".\n"
        sentences = re.split(r"(?<=[.!?])\s+|\.\n", text)
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i : i + self.max_sentences_per_chunk]
            chunk = " ".join(chunk_sentences)
            chunks.append(chunk.strip())

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self, separators: list[str] | None = None, chunk_size: int = 500
    ) -> None:
        self.separators = (
            self.DEFAULT_SEPARATORS if separators is None else list(separators)
        )
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: if text fits in chunk_size, return it
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []

        # If no separators left, split by character
        if not remaining_separators:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunk = current_text[i : i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # If separator is empty, split by character
        if separator == "":
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunk = current_text[i : i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks

        # Split by current separator
        parts = current_text.split(separator)

        chunks = []
        current_chunk = ""

        for part in parts:
            # Check if adding this part (with separator if not first) would exceed chunk_size
            separator_prefix = separator if current_chunk else ""
            potential_chunk = current_chunk + separator_prefix + part

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                # Start new chunk with current part
                # If part itself is too big, recursively split with next separator
                if len(part) > self.chunk_size:
                    sub_chunks = self._split(part, next_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Create chunkers
        fixed_size = FixedSizeChunker(chunk_size=chunk_size, overlap=0)
        by_sentences = SentenceChunker(max_sentences_per_chunk=3)
        recursive = RecursiveChunker(chunk_size=chunk_size)

        # Get chunks from each strategy
        fixed_chunks = fixed_size.chunk(text)
        sentence_chunks = by_sentences.chunk(text)
        recursive_chunks = recursive.chunk(text)

        def compute_stats(chunks: list[str]) -> dict:
            if not chunks:
                return {"count": 0, "avg_length": 0.0, "chunks": []}
            total_len = sum(len(c) for c in chunks)
            return {
                "count": len(chunks),
                "avg_length": total_len / len(chunks),
                "chunks": chunks,
            }

        return {
            "fixed_size": compute_stats(fixed_chunks),
            "by_sentences": compute_stats(sentence_chunks),
            "recursive": compute_stats(recursive_chunks),
        }
