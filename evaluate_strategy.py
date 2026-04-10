#!/usr/bin/env python3
"""Evaluate retrieval strategy using 5 criteria from README."""

from process_products import main as process_data
from benchmark_queries import BENCHMARK_QUERIES


def evaluate_retrieval_precision(store, queries, top_k=3):
    """
    1. Top-k có chứa chunk thật sự liên quan không?

    Evaluates if the retrieved top-k chunks actually contain relevant content.
    Checks for keyword presence in retrieved results.
    """
    results = []
    for query in queries:
        retrieved = store.search(query["query"], top_k=top_k)

        # Check if any retrieved chunk contains expected keywords
        found_keywords = set()
        for result in retrieved:
            content = result.get("content", "").lower()
            for keyword in query["expected_keywords"]:
                if keyword.lower() in content:
                    found_keywords.add(keyword)

        precision = len(found_keywords) / len(query["expected_keywords"])
        results.append(
            {
                "query_id": query["id"],
                "query": query["query"],
                "precision": precision,
                "keywords_found": list(found_keywords),
                "keywords_expected": query["expected_keywords"],
                "retrieved_brands": [
                    r.get("metadata", {}).get("brand", "unknown") for r in retrieved
                ],
            }
        )

    return results


def evaluate_chunk_coherence(store, queries):
    """
    2. Chunk có giữ được ý trọn vẹn không?

    Evaluates if chunks maintain semantic coherence by checking:
    - Chunk size distribution
    - Whether chunks break in the middle of sentences/sections
    """
    results = []
    for query in queries:
        retrieved = store.search(query["query"], top_k=3)

        coherence_scores = []
        for result in retrieved:
            content = result.get("content", "")

            # Check if chunk ends abruptly
            ends_cleanly = (
                content.endswith(".")
                or content.endswith("?")
                or content.endswith("!")
                or content.endswith("\n")
                or content.endswith(":")
            )

            # Check if chunk starts with complete content (not mid-sentence)
            starts_cleanly = content[0].isupper() if content else False

            score = (1.0 if ends_cleanly else 0.5) + (1.0 if starts_cleanly else 0.5)
            coherence_scores.append(score / 2.0)

        avg_coherence = (
            sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
        )
        results.append(
            {
                "query_id": query["id"],
                "coherence_score": avg_coherence,
                "chunk_count": len(retrieved),
            }
        )

    return results


def evaluate_metadata_utility(store, queries):
    """
    3. search_with_filter() có giúp tăng độ chính xác không?

    Tests if metadata filtering improves search precision by comparing
    filtered vs unfiltered results.
    """
    results = []
    for query in queries:
        # Unfiltered search
        unfiltered = store.search(query["query"], top_k=5)
        unfiltered_brand_hits = sum(
            1
            for r in unfiltered
            if r.get("metadata", {}).get("brand") == query["expected_brand"]
        )

        # Filtered search by expected brand
        filtered = store.search_with_filter(
            query["query"], top_k=5, metadata_filter={"brand": query["expected_brand"]}
        )

        # Calculate precision improvement
        unfiltered_precision = (
            unfiltered_brand_hits / len(unfiltered) if unfiltered else 0
        )
        filtered_precision = (
            1.0 if filtered else 0
        )  # All filtered results match expected brand

        improvement = filtered_precision - unfiltered_precision

        results.append(
            {
                "query_id": query["id"],
                "unfiltered_precision": unfiltered_precision,
                "filtered_precision": filtered_precision,
                "improvement": improvement,
                "filtered_results_count": len(filtered),
            }
        )

    return results


def evaluate_grounding_quality(store, queries):
    """
    4. Agent trả lời có dựa trên retrieved context không?

    Evaluates if the retrieved content contains sufficient information
    to ground an agent's response. Checks content completeness.
    """
    results = []
    for query in queries:
        retrieved = store.search(query["query"], top_k=3)

        # Check if retrieved content is substantial enough
        total_content_length = sum(len(r.get("content", "")) for r in retrieved)
        avg_content_length = total_content_length / len(retrieved) if retrieved else 0

        # Check if metadata is present (important for grounding)
        metadata_completeness = []
        for result in retrieved:
            metadata = result.get("metadata", {})
            has_brand = metadata.get("brand", "unknown") != "unknown"
            has_product = metadata.get("product", "unknown") != "unknown"
            has_category = metadata.get("category", "unknown") != "unknown"
            metadata_completeness.append((has_brand + has_product + has_category) / 3.0)

        avg_metadata = (
            sum(metadata_completeness) / len(metadata_completeness)
            if metadata_completeness
            else 0
        )

        # Grounding score combines content length and metadata completeness
        grounding_score = min(1.0, avg_content_length / 500) * 0.5 + avg_metadata * 0.5

        results.append(
            {
                "query_id": query["id"],
                "grounding_score": grounding_score,
                "avg_content_length": avg_content_length,
                "metadata_completeness": avg_metadata,
            }
        )

    return results


def evaluate_data_strategy_impact(store, queries):
    """
    5. Bộ tài liệu phù hợp với benchmark queries không?

    Evaluates overall data coverage and appropriateness for the queries.
    Checks if the collection has content for all relevant brands and categories.
    """
    # Get collection statistics
    total_docs = store.get_collection_size()

    # Check coverage across brands
    brand_coverage = {}
    for brand in ["schwind", "melag", "bvi"]:
        brand_results = store.search_with_filter(
            "medical products", top_k=total_docs, metadata_filter={"brand": brand}
        )
        brand_coverage[brand] = len(brand_results)

    # Check coverage across categories
    categories = set(q["category"] for q in queries)
    category_coverage = {}
    for category in categories:
        # Try to find content for this category
        sample_query = next(q for q in queries if q["category"] == category)
        results = store.search(sample_query["query"], top_k=10)
        category_coverage[category] = len(results)

    # Calculate overall coverage score
    brand_balance = (
        min(brand_coverage.values()) / max(brand_coverage.values())
        if brand_coverage
        else 0
    )

    return {
        "total_documents": total_docs,
        "brand_coverage": brand_coverage,
        "category_coverage": category_coverage,
        "brand_balance_score": brand_balance,
    }


def main():
    # Load or create store
    print("Loading/processing data...")
    store = process_data()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Strategy: RecursiveChunker(chunk_size=768, overlap=100)")
    print(f"Separators: [H2, H3, paragraphs, lines, sentences, words]")
    print(f"Embedding: OllamaEmbedder(nomic-embed-text-v2-moe)")
    print(f"Collection: medical_products")
    print("=" * 60)

    # Run all evaluations
    print("\n" + "-" * 60)
    print("1. RETRIEVAL PRECISION (Top-k relevance)")
    print("-" * 60)
    precision_results = evaluate_retrieval_precision(store, BENCHMARK_QUERIES)
    for result in precision_results:
        print(f"\nQuery {result['query_id']}: {result['query']}")
        print(f"  Precision: {result['precision']:.2%}")
        print(f"  Keywords found: {', '.join(result['keywords_found'])}")
        print(f"  Brands retrieved: {', '.join(set(result['retrieved_brands']))}")

    print("\n" + "-" * 60)
    print("2. CHUNK COHERENCE")
    print("-" * 60)
    coherence_results = evaluate_chunk_coherence(store, BENCHMARK_QUERIES)
    avg_coherence = sum(r["coherence_score"] for r in coherence_results) / len(
        coherence_results
    )
    print(f"Average coherence score: {avg_coherence:.2%}")
    for result in coherence_results:
        print(f"  Query {result['query_id']}: {result['coherence_score']:.2%}")

    print("\n" + "-" * 60)
    print("3. METADATA FILTER UTILITY")
    print("-" * 60)
    filter_results = evaluate_metadata_utility(store, BENCHMARK_QUERIES)
    for result in filter_results:
        print(f"\nQuery {result['query_id']}:")
        print(f"  Unfiltered precision: {result['unfiltered_precision']:.2%}")
        print(f"  Filtered precision: {result['filtered_precision']:.2%}")
        print(f"  Improvement: {result['improvement']:+.2%}")

    print("\n" + "-" * 60)
    print("4. GROUNDING QUALITY")
    print("-" * 60)
    grounding_results = evaluate_grounding_quality(store, BENCHMARK_QUERIES)
    avg_grounding = sum(r["grounding_score"] for r in grounding_results) / len(
        grounding_results
    )
    print(f"Average grounding score: {avg_grounding:.2%}")
    for result in grounding_results:
        print(
            f"  Query {result['query_id']}: {result['grounding_score']:.2%} "
            f"(content: {result['avg_content_length']:.0f} chars, "
            f"metadata: {result['metadata_completeness']:.0%})"
        )

    print("\n" + "-" * 60)
    print("5. DATA STRATEGY IMPACT")
    print("-" * 60)
    data_results = evaluate_data_strategy_impact(store, BENCHMARK_QUERIES)
    print(f"Total documents in collection: {data_results['total_documents']}")
    print(f"\nBrand coverage:")
    for brand, count in data_results["brand_coverage"].items():
        print(f"  {brand}: {count} documents")
    print(f"Brand balance score: {data_results['brand_balance_score']:.2%}")
    print(f"\nCategory coverage:")
    for category, count in data_results["category_coverage"].items():
        print(f"  {category}: {count} results")

    print("\n" + "=" * 60)
    print("BASIC SEARCH TEST")
    print("=" * 60)
    # Run a basic search test
    for query in BENCHMARK_QUERIES:
        print(f"\nQuery {query['id']}: {query['query']}")
        results = store.search(query["query"], top_k=5)

        for i, result in enumerate(results[:3], 1):
            metadata = result.get("metadata", {})
            score = result.get("score", 0.0)
            print(
                f"  {i}. Score: {score:.4f} | "
                f"Brand: {metadata.get('brand', 'unknown')} | "
                f"Lang: {metadata.get('language', 'unknown')}"
            )

        # Test with filter
        filtered = store.search_with_filter(
            query["query"], top_k=3, metadata_filter={"brand": query["expected_brand"]}
        )
        print(
            f"  With brand filter ({query['expected_brand']}): {len(filtered)} results"
        )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
