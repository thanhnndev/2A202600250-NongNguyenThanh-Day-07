#!/usr/bin/env python3
"""Process cleaned product files for Phase 2."""

import yaml
from pathlib import Path
from src import RecursiveChunker, OllamaEmbedder, EmbeddingStore, Document

# Strategy parameters
CHUNK_SIZE = 768
OVERLAP = 100
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]


def main():
    # Initialize components
    chunker = RecursiveChunker(separators=SEPARATORS, chunk_size=CHUNK_SIZE)
    embedder = OllamaEmbedder()
    store = EmbeddingStore(collection_name="medical_products", embedding_fn=embedder)

    # Process files
    products_dir = Path("products_cleaned")
    processed = 0
    total_chunks = 0

    for file_path in products_dir.rglob("*.md"):
        # Read and parse
        content = file_path.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            metadata = yaml.safe_load(parts[1])
            body = parts[2].strip()
        else:
            metadata = {}
            body = content

        # Chunk the content
        chunks = chunker.chunk(body)

        # Create documents for each chunk
        for i, chunk in enumerate(chunks):
            doc = Document(
                id=f"{metadata.get('product', 'unknown')}_{i}",
                content=chunk,
                metadata={
                    "brand": metadata.get("brand", "unknown"),
                    "category": metadata.get("category", "unknown"),
                    "language": metadata.get("language", "unknown"),
                    "product": metadata.get("product", "unknown"),
                    "original_file": metadata.get("original_file", str(file_path)),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            store.add_documents([doc])
            total_chunks += 1

        processed += 1
        print(f"Processed: {file_path} -> {len(chunks)} chunks")

    print(f"\n{'=' * 50}")
    print(f"Files processed: {processed}")
    print(f"Total chunks: {total_chunks}")
    print(f"Collection size: {store.get_collection_size()}")

    return store


if __name__ == "__main__":
    store = main()
