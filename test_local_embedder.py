#!/usr/bin/env python3
"""
Test script for LocalEmbedder with multilingual model.

Usage (after installing sentence-transformers):
    .venv/bin/pip install sentence-transformers
    .venv/bin/python test_local_embedder.py

This will download the model on first run (~471MB).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("TESTING LOCAL EMBEDDER WITH MULTILINGUAL MODEL")
print("=" * 60)

# Check if sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer

    print("✅ sentence-transformers is installed")
except ImportError:
    print("❌ sentence-transformers is NOT installed")
    print("\nPlease install it first:")
    print("  .venv/bin/pip install sentence-transformers")
    sys.exit(1)

# Test models
MODELS_TO_TEST = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "description": "Recommended multilingual model (384 dims, 471MB)",
        "primary": True,
    },
    {
        "name": "all-MiniLM-L6-v2",
        "description": "Current default (English-only, 80MB)",
        "primary": False,
    },
]

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for model_info in MODELS_TO_TEST:
    model_name = model_info["name"]
    print(f"\n📦 Testing: {model_name}")
    print(f"   Description: {model_info['description']}")

    try:
        # Load model (this will download on first run)
        print(f"   Loading model... (may download on first run)")
        embedder = SentenceTransformer(model_name)

        # Test texts
        test_texts = [
            "laser eye surgery technology",
            "công nghệ phẫu thuật laser mắt",
            "medical sterilization equipment",
            "thiết bị tiệt trùng y tế",
        ]

        # Generate embeddings
        embeddings = embedder.encode(test_texts, normalize_embeddings=True)

        print(f"   ✅ Successfully generated {len(embeddings)} embeddings")
        print(f"   Dimensions: {len(embeddings[0])}")

        # Test cross-lingual similarity
        if len(embeddings) >= 2:
            from src.chunking import compute_similarity

            # EN vs VI similarity
            en_idx = 0  # "laser eye surgery technology"
            vi_idx = 1  # "công nghệ phẫu thuật laser mắt"

            similarity = compute_similarity(
                embeddings[en_idx].tolist(), embeddings[vi_idx].tolist()
            )

            print(f"   Cross-lingual similarity (EN↔VI): {similarity:.4f}")

            if similarity > 0.7:
                print(f"   🎉 Excellent! Model handles cross-lingual well")
            elif similarity > 0.5:
                print(f"   ✅ Good cross-lingual alignment")
            else:
                print(f"   ⚠️ Weak cross-lingual alignment")

        if model_info["primary"]:
            print(f"   ⭐ RECOMMENDED MODEL")

    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)

print("""
Model Recommendations for Medical Products:

1. paraphrase-multilingual-MiniLM-L12-v2 ⭐
   - Best for: English + Vietnamese mixed data
   - Dimensions: 384
   - Size: 471MB
   - Performance: Excellent cross-lingual retrieval
   
2. all-MiniLM-L6-v2 (current default)
   - Best for: English-only data
   - Dimensions: 384
   - Size: 80MB
   - Performance: Won't work well with Vietnamese

To use the recommended model:
  1. Install: .venv/bin/pip install sentence-transformers
  2. Update src/embeddings.py: LOCAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
  3. Run: from src import LocalEmbedder; e = LocalEmbedder()
""")

print("=" * 60)
