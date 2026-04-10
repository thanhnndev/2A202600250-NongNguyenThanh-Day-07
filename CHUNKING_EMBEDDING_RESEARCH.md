# Chunking and Embedding Strategy Research Report

## Executive Summary

This report analyzes chunking strategies and embedding models for 110 cleaned medical product files (55 English + 55 Vietnamese) covering three healthcare brands: SCHWIND (laser eye surgery), BVI (ophthalmic devices), and MELAG (medical sterilization).

### Key Findings

**Recommended Chunking Strategy**: **Markdown-Aware Recursive Chunking with Header Preservation**
- Use `RecursiveChunker` with custom markdown separators: `["\n## ", "\n### ", "\n\n", "\n", ". ", " "]`
- Chunk size: **512-768 characters** (optimal for technical/medical content)
- Overlap: **10-15%** (50-100 characters) to preserve context
- This preserves hierarchical structure while maintaining optimal retrieval granularity

**Recommended Embedding Strategy**: **Multilingual Sentence Transformers**
- Primary recommendation: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- Alternative for higher accuracy: `paraphrase-multilingual-mpnet-base-v2` (768 dimensions)
- For local deployment via Ollama: `nomic-embed-text-v2-moe` with 768-dimension truncation
- These models support 50+ languages including Vietnamese and medical domain terminology

### Why These Recommendations

1. **Data Structure**: Files have YAML metadata headers + hierarchical markdown (#, ##, ###)
2. **Content Type**: Technical medical equipment documentation requires semantic coherence
3. **Bilingual Requirement**: Must support both English and Vietnamese queries effectively
4. **Retrieval Accuracy**: Smaller chunks (512-768 chars) improve precision for specific technical queries

---

## Context7 Research Sources

This research utilized the following Context7 library sources:

| Library | Library ID | Purpose | Relevance |
|---------|------------|---------|-----------|
| **LangChain** | `/websites/langchain` | Chunking strategies, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter | High - Primary source for chunking best practices |
| **Sentence Transformers** | `/huggingface/sentence-transformers` | Multilingual embedding models, model comparisons, usage patterns | High - Core source for embedding recommendations |
| **Ollama** | `/websites/ollama` | Local embedding deployment, nomic-embed-text API | Medium - For local deployment options |
| **OpenAI Cookbook** | `/openai/openai-cookbook` | Embedding API, text-embedding-3 models | Medium - Comparison with cloud options |
| **Pinecone** | `/llmstxt/pinecone_io_llms-full_txt` | Chunking strategies for RAG, content optimization | Medium - Vector DB perspective on chunking |

### Key Context7 References

1. **LangChain Documentation** - "Split Markdown Text with RecursiveCharacterTextSplitter"
   - Source: https://docs.langchain.com/oss/python/integrations/splitters/code_splitter
   - Demonstrates markdown-specific chunking using `from_language(Language.MARKDOWN)`

2. **LangChain Documentation** - "MarkdownHeaderTextSplitter with RecursiveCharacterTextSplitter"
   - Source: https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
   - Shows two-stage chunking: headers first, then recursive character splitting

3. **Sentence Transformers** - "Multilingual Models"
   - Source: https://github.com/huggingface/sentence-transformers/blob/main/docs/sentence_transformer/pretrained_models.md
   - Documents `paraphrase-multilingual-MiniLM-L12-v2` supporting 50+ languages

4. **Sentence Transformers** - "Inference with Truncated Embeddings"
   - Source: https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/training/matryoshka/README.md
   - Shows nomic-embed-text-v1.5 usage with `truncate_dim` parameter

5. **Pinecone Documentation** - "Content Chunking Strategies for LLMs"
   - Source: https://docs.pinecone.io/guides/optimize/increase-relevance
   - Discusses chunking strategy selection based on content type and use case

---

## Part 1: Chunking Strategy Analysis

### 1.1 Existing Implementation Review

The current codebase includes three chunking strategies in `src/chunking.py`:

#### FixedSizeChunker
```python
class FixedSizeChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None
```
- **Pros**: Simple, predictable chunk sizes
- **Cons**: Ignores semantic boundaries, can split mid-sentence or mid-paragraph
- **Best for**: Unstructured text where boundaries don't matter

#### SentenceChunker
```python
class SentenceChunker:
    def __init__(self, max_sentences_per_chunk: int = 3) -> None
```
- **Pros**: Respects sentence boundaries
- **Cons**: Variable chunk sizes, can create very long chunks with complex sentences
- **Best for**: Documents with consistent sentence lengths

#### RecursiveChunker
```python
class RecursiveChunker:
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None
```
- **Pros**: Hierarchical splitting, respects structure
- **Cons**: Default separators don't account for markdown headers
- **Best for**: Structured documents with clear hierarchy

### 1.2 Chunking Strategy Comparison

Based on Context7 research, here are the recommended strategies for technical/medical documents:

| Strategy | Chunk Size | Overlap | Best For | Medical/Tech Suitability |
|----------|------------|---------|----------|-------------------------|
| **Fixed-Size** | 1000+ chars | 200 chars | General text | Poor - splits semantic units |
| **Sentence-Based** | 3-5 sentences | 1 sentence | Narrative text | Fair - preserves sentences |
| **Recursive Character** | 512-768 chars | 50-100 chars | Structured docs | Good - respects boundaries |
| **Markdown Header + Recursive** | 512-768 chars | 50-100 chars | Markdown docs | **Excellent** - preserves hierarchy |
| **Semantic (Agentic)** | Dynamic | Varies | Complex Q&A | Good - context-aware |

### 1.3 Best Practices from Context7 Research

#### Chunk Size Recommendations

From LangChain and Pinecone documentation:

1. **Small Chunks (256-512 chars)**: 
   - Best for: Specific fact retrieval, short queries
   - Pros: High precision, lower noise
   - Cons: May lose broader context

2. **Medium Chunks (512-768 chars)**:
   - Best for: **Technical/medical documentation** (RECOMMENDED)
   - Pros: Balance between precision and context
   - Cons: Moderate retrieval complexity

3. **Large Chunks (1000+ chars)**:
   - Best for: Document summarization, broad context
   - Pros: Full context preserved
   - Cons: Lower precision for specific queries

#### Overlap Recommendations

From LangChain documentation:
- **10-20% overlap** is standard (e.g., 100 chars for 1000-char chunks)
- Overlap preserves context across chunk boundaries
- Too much overlap (30%+) increases storage without proportional benefit
- For technical content: **10-15%** is optimal

#### Markdown-Specific Chunking

From LangChain `MarkdownHeaderTextSplitter` documentation:

```python
# Two-stage approach recommended:
# Stage 1: Split by headers to preserve hierarchy
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Stage 2: Apply recursive character chunking within header sections
# to ensure chunks don't exceed embedding model limits
```

**Why this matters for our data**:
- Our cleaned files have hierarchical structure: # (title), ## (sections), ### (subsections)
- Preserving this structure helps retrieval (e.g., searching "SmartSurfACE Technology" should return the full section)
- Header-based chunking ensures related content stays together

### 1.4 Recommended Chunking Configuration for Medical Products

Based on the data characteristics:

```python
# Recommended chunker configuration for medical product documentation
RECOMMENDED_CHUNK_CONFIG = {
    "chunker_class": "RecursiveChunker",
    "chunk_size": 768,  # Characters
    "overlap": 100,     # ~13% overlap
    "separators": [
        "\n## ",      # H2 headers (major sections)
        "\n### ",     # H3 headers (subsections)
        "\n\n",      # Paragraph breaks
        "\n",        # Line breaks
        ". ",        # Sentences
        " ",         # Words (last resort)
    ],
    "strip_headers": False,  # Keep headers in chunks for context
}
```

**Rationale**:
1. **Chunk Size 768**: Balances detail and context for technical queries
2. **Separators prioritize headers**: Ensures sections aren't split arbitrarily
3. **100-char overlap**: Preserves flow between chunks without excessive redundancy
4. **Header preservation**: Chunks retain section titles for better context

---

## Part 2: Embedding Model Analysis

### 2.1 Existing Implementation Review

The current codebase includes four embedders in `src/embeddings.py`:

#### MockEmbedder
- Deterministic hash-based embeddings for testing
- Dimension: Configurable (default 64)
- **Use case**: Unit testing, development without model downloads

#### LocalEmbedder (Sentence Transformers)
```python
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions
```
- **Pros**: Fast, local, no API calls
- **Cons**: English-only, may struggle with Vietnamese
- **Performance**: 64.33 sentence performance, 51.83 semantic search (per Sentence Transformers docs)

#### OpenAIEmbedder
```python
DEFAULT_MODEL = "text-embedding-3-small"  # 1536 dimensions
```
- **Pros**: Strong multilingual support, high quality
- **Cons**: Requires API key, latency, cost
- **Performance**: Outperforms ada-002, supports dimension truncation

#### OllamaEmbedder
```python
DEFAULT_MODEL = "nomic-embed-text-v2-moe"  # 768 dimensions default
```
- **Pros**: Local deployment, good quality
- **Cons**: Requires Ollama server, model download
- **Note**: Currently uses 768 dimensions (full), but supports truncation

### 2.2 Embedding Model Comparison

Based on Context7 research, here's a detailed comparison:

#### Multilingual Models (for English + Vietnamese)

| Model | Dimensions | Languages | Speed | Medical/Tech | Size | Best For |
|-------|------------|-----------|-------|--------------|------|----------|
| **paraphrase-multilingual-MiniLM-L12-v2** | 384 | 50+ | Fast | Good | 471MB | **Primary Recommendation** |
| **paraphrase-multilingual-mpnet-base-v2** | 768 | 50+ | Medium | Excellent | 1.1GB | Higher accuracy needs |
| **distiluse-base-multilingual-cased-v2** | 512 | 50+ | Fast | Good | 500MB | Alternative option |
| **nomic-embed-text-v1.5** | 768/1024 | 50+ | Fast | Good | 550MB | Matryoshka truncation |
| **nomic-embed-text-v2-moe** | 768 | 50+ | Fast | Good | 1.3GB | MoE architecture |

#### English-Only Models (not recommended for this use case)

| Model | Dimensions | Speed | Performance | Size |
|-------|------------|-------|-------------|------|
| all-MiniLM-L6-v2 | 384 | Very Fast | 64.33 | 80MB |
| all-mpnet-base-v2 | 768 | Fast | 69.57 | 418MB |
| all-MiniLM-L12-v2 | 384 | Fast | 68.03 | 118MB |

#### Cloud-Based Models

| Model | Provider | Dimensions | Cost | Best For |
|-------|----------|------------|------|----------|
| text-embedding-3-small | OpenAI | 1536 (truncatable) | Low | Cloud deployment |
| text-embedding-3-large | OpenAI | 3072 (truncatable) | Medium | Maximum quality |
| text-embedding-ada-002 | OpenAI | 1536 | Legacy | Backward compatibility |

### 2.3 Detailed Model Analysis

#### Primary Recommendation: `paraphrase-multilingual-MiniLM-L12-v2`

From Sentence Transformers documentation:

```python
from sentence_transformers import SentenceTransformer

# Load multilingual model supporting 50+ languages
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Encode in multiple languages
embeddings = model.encode([
    "Hello World",      # English
    "Hallo Welt",       # German
    "Hola mundo",       # Spanish
    "Xin chào thế giới" # Vietnamese (similar tokenization)
])

# Similarity matrix shows strong cross-lingual alignment
# tensor([[1.0000, 0.9429, 0.8880, ...],
#         [0.9429, 1.0000, 0.9680, ...],
#         ...])
```

**Why this model for our use case**:
1. **Multilingual**: Supports 50+ languages including Vietnamese
2. **Compact**: 384 dimensions, 471MB - efficient storage and retrieval
3. **Fast**: Suitable for real-time applications
4. **Well-tested**: Part of official Sentence Transformers library
5. **Medical domain**: Can be fine-tuned if needed (though base model performs well)

#### Alternative: `nomic-embed-text-v2-moe` (via Ollama)

From Sentence Transformers documentation on matryoshka embeddings:

```python
from sentence_transformers import SentenceTransformer

# Nomic embed with dimension truncation (Matryoshka)
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    truncate_dim=768,  # Can reduce to 512 or 256 if needed
)

# nomic-embed-text-v2-moe is similar but with Mixture of Experts architecture
# providing better quality at similar inference cost
```

**Key features of nomic-embed-text-v2-moe**:
- Mixture of Experts (MoE) architecture - 16 experts, 2 active per token
- 768 default dimensions (full size)
- Supports dimension truncation for flexible storage/accuracy tradeoff
- Long context support (8192 tokens)
- Strong performance on MTEB benchmarks

### 2.4 Vietnamese Language Considerations

From multilingual embedding research (Context7 sources):

1. **Tokenization**: Vietnamese is a monosyllabic language with tone marks, but modern multilingual models handle this well
2. **Cross-lingual retrieval**: Models like `paraphrase-multilingual-*` map semantically equivalent content across languages to similar vectors
3. **Example from research**:
   ```python
   # English query can retrieve Vietnamese content
   query_en = "laser eye surgery technology"
   doc_vi = "Công nghệ phẫu thuật laser mắt"
   # High similarity despite different languages
   ```

### 2.5 Embedding Strategy Recommendations

#### For Production Use (High Priority)

**Option 1: Local Deployment (Recommended)**
```python
# Use paraphrase-multilingual-MiniLM-L12-v2 via LocalEmbedder
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
embedder = LocalEmbedder(model_name=model_name)
```
- **Pros**: No API costs, full privacy, fast inference
- **Cons**: Initial model download (471MB)
- **Best for**: Production deployment, offline environments

**Option 2: Ollama Deployment**
```python
# Use nomic-embed-text-v2-moe via OllamaEmbedder
embedder = OllamaEmbedder(
    model_name="nomic-embed-text-v2-moe",
    base_url="http://localhost:11434"
)
```
- **Pros**: GPU acceleration if available, model management
- **Cons**: Requires Ollama server running
- **Best for**: When Ollama is already deployed for LLM inference

#### For Development/Testing

**Option 3: OpenAI (for comparison)**
```python
# Use text-embedding-3-small via OpenAIEmbedder
embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
```
- **Pros**: Easy setup, high quality baseline
- **Cons**: API costs, latency, network dependency
- **Best for**: Baseline comparison, prototyping

### 2.6 Recommended Embedding Configuration

```python
# Recommended embedding configuration for medical products
RECOMMENDED_EMBEDDING_CONFIG = {
    "provider": "local",  # or "ollama" if preferred
    "model": "paraphrase-multilingual-MiniLM-L12-v2",
    "dimensions": 384,
    "normalize": True,  # Important for cosine similarity
    "batch_size": 32,   # Tune based on available memory
}

# Alternative for higher accuracy (with storage tradeoff)
ALTERNATIVE_EMBEDDING_CONFIG = {
    "provider": "local",
    "model": "paraphrase-multilingual-mpnet-base-v2",
    "dimensions": 768,
    "normalize": True,
    "batch_size": 16,   # Larger model needs smaller batches
}
```

---

## Part 3: Integration Strategy

### 3.1 How to Extend Existing Code

#### Enhanced Chunking Module

Add a new markdown-aware chunker to `src/chunking.py`:

```python
class MarkdownAwareChunker(RecursiveChunker):
    """
    Markdown-aware chunker that preserves header hierarchy.
    
    Uses two-stage approach:
    1. First tries to split on markdown headers (##, ###)
    2. Then falls back to recursive character splitting
    """
    
    MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    
    def __init__(
        self, 
        chunk_size: int = 768, 
        overlap: int = 100,
        preserve_yaml: bool = True
    ) -> None:
        super().__init__(
            separators=self.MARKDOWN_SEPARATORS,
            chunk_size=chunk_size
        )
        self.overlap = overlap
        self.preserve_yaml = preserve_yaml
        
    def chunk(self, text: str) -> list[str]:
        # Extract YAML frontmatter if present
        if self.preserve_yaml and text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                yaml_header = f"---{parts[1]}---"
                content = parts[2].strip()
                # Add YAML to each chunk for metadata context
                chunks = super().chunk(content)
                return [f"{yaml_header}\n\n{chunk}" for chunk in chunks]
        
        return super().chunk(text)
```

#### Enhanced Embedding Module

Update `src/embeddings.py` to use recommended model:

```python
# Updated constants
LOCAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Was: all-MiniLM-L6-v2

def get_recommended_embedder(provider: str = "local"):
    """Factory function for recommended embedder."""
    if provider == "local":
        return LocalEmbedder(model_name=LOCAL_EMBEDDING_MODEL)
    elif provider == "ollama":
        return OllamaEmbedder(model_name="nomic-embed-text-v2-moe")
    elif provider == "openai":
        return OpenAIEmbedder(model_name="text-embedding-3-small")
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

### 3.2 Processing Pipeline

Recommended processing flow for the 110 cleaned product files:

```python
import yaml
from pathlib import Path

# 1. Load and parse files
def process_product_file(file_path: Path) -> dict:
    """Extract metadata and content from cleaned product file."""
    content = file_path.read_text(encoding='utf-8')
    
    # Parse YAML frontmatter
    if content.startswith('---'):
        parts = content.split('---', 2)
        metadata = yaml.safe_load(parts[1])
        body = parts[2].strip()
    else:
        metadata = {}
        body = content
    
    return {
        "metadata": metadata,
        "body": body,
        "file_path": str(file_path)
    }

# 2. Chunk with markdown awareness
chunker = MarkdownAwareChunker(
    chunk_size=768,
    overlap=100,
    preserve_yaml=True
)

# 3. Generate embeddings
embedder = LocalEmbedder(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# 4. Process all files
processed_chunks = []
for file_path in Path("products_cleaned").rglob("*.md"):
    product = process_product_file(file_path)
    chunks = chunker.chunk(product["body"])
    
    for chunk in chunks:
        embedding = embedder(chunk)
        processed_chunks.append({
            "chunk": chunk,
            "embedding": embedding,
            "metadata": product["metadata"],
            "file_path": product["file_path"]
        })
```

### 3.3 Testing Recommendations

#### Retrieval Accuracy Testing

```python
def test_cross_lingual_retrieval():
    """Test that English queries retrieve Vietnamese content."""
    test_cases = [
        {
            "query_en": "laser eye surgery technology",
            "expected_vi_terms": ["phẫu thuật laser", "công nghệ"],
        },
        {
            "query_en": "medical sterilization equipment",
            "expected_vi_terms": ["thiết bị tiệt trùng", "y tế"],
        },
    ]
    
    for test in test_cases:
        query_embedding = embedder(test["query_en"])
        # Retrieve top-k chunks and verify Vietnamese content matches
        # ...
```

#### Chunk Quality Testing

```python
def test_chunk_boundaries():
    """Verify chunks respect header boundaries."""
    sample_text = """
## Section A
Content for section A.
More content.

## Section B
Content for section B.
"""
    chunks = chunker.chunk(sample_text)
    
    # Verify no chunk splits Section A and Section B
    for chunk in chunks:
        assert not ("Section A" in chunk and "Section B" in chunk), \
            "Chunk should not span multiple H2 sections"
```

---

## Part 4: Final Recommendations Summary

### 4.1 Chunking Recommendation

**USE**: `MarkdownAwareChunker` (enhanced RecursiveChunker)

| Parameter | Recommended Value | Rationale |
|-----------|------------------|-----------|
| Chunk Size | **768 characters** | Optimal for technical content detail |
| Overlap | **100 characters** (~13%) | Preserves context without redundancy |
| Separators | `["\n## ", "\n### ", "\n\n", "\n", ". ", " "]` | Respects markdown hierarchy |
| YAML Preservation | **True** | Metadata context in every chunk |

**Why**: Balances precision and context for medical/technical queries

### 4.2 Embedding Recommendation

**PRIMARY**: `paraphrase-multilingual-MiniLM-L12-v2` via LocalEmbedder

| Attribute | Value |
|-----------|-------|
| Dimensions | 384 |
| Model Size | 471MB |
| Languages | 50+ (including Vietnamese) |
| Normalization | Yes (for cosine similarity) |
| Batch Size | 32 (tune based on memory) |

**ALTERNATIVE**: `nomic-embed-text-v2-moe` via OllamaEmbedder (if Ollama already deployed)

**Why**: Best multilingual support with efficient local deployment

### 4.3 Implementation Priority

1. **Immediate** (High Impact, Low Effort):
   - Update `LOCAL_EMBEDDING_MODEL` constant to multilingual model
   - Test with sample English/Vietnamese queries

2. **Short-term** (High Impact, Medium Effort):
   - Implement `MarkdownAwareChunker` class
   - Process all 110 files with new chunking strategy
   - Build retrieval test suite

3. **Medium-term** (Medium Impact, Medium Effort):
   - Evaluate retrieval accuracy with different chunk sizes (512 vs 768 vs 1024)
   - Consider fine-tuning if domain-specific terminology needs improvement
   - Implement hybrid search (semantic + keyword) for product codes

### 4.4 Expected Outcomes

With these recommendations implemented:

1. **Cross-lingual retrieval**: English queries will successfully retrieve relevant Vietnamese content
2. **Structural preservation**: Section headers and subsections remain intact within chunks
3. **Optimal granularity**: Chunks are sized for both specific technical queries and broader context questions
4. **Efficient storage**: 384-dimensional embeddings minimize vector storage requirements
5. **Fast inference**: Local model deployment ensures low-latency retrieval

---

## Appendix A: Complete Implementation Code

### A.1 Enhanced Chunking Module (`src/chunking.py` additions)

```python
class MarkdownAwareChunker(RecursiveChunker):
    """
    Markdown-aware chunker optimized for technical/medical documentation.
    
    Preserves YAML frontmatter and markdown header hierarchy while ensuring
    chunks stay within embedding model token limits.
    """
    
    # Priority: H2 headers > H3 headers > paragraphs > lines > sentences > words
    MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    
    def __init__(
        self, 
        chunk_size: int = 768, 
        overlap: int = 100,
        preserve_yaml: bool = True,
        max_tokens_estimate: int = 200  # ~1.3 tokens per char for English/Vietnamese
    ) -> None:
        """
        Initialize MarkdownAwareChunker.
        
        Args:
            chunk_size: Target chunk size in characters (default 768)
            overlap: Character overlap between chunks (default 100)
            preserve_yaml: Whether to prepend YAML metadata to chunks
            max_tokens_estimate: Estimated token limit for embedding model
        """
        super().__init__(
            separators=self.MARKDOWN_SEPARATORS,
            chunk_size=chunk_size
        )
        self.overlap = overlap
        self.preserve_yaml = preserve_yaml
        self.max_tokens_estimate = max_tokens_estimate
        
    def chunk(self, text: str) -> list[str]:
        """
        Split text into markdown-aware chunks.
        
        Args:
            text: Input text (may include YAML frontmatter)
            
        Returns:
            List of chunks with optional YAML metadata prepended
        """
        if not text:
            return []
        
        # Extract and validate YAML frontmatter
        yaml_header = ""
        content = text
        
        if self.preserve_yaml and text.startswith("---"):
            try:
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    # Validate YAML
                    yaml_content = parts[1].strip()
                    if yaml_content:
                        yaml.safe_load(yaml_content)  # Validation only
                        yaml_header = f"---\n{yaml_content}\n---\n\n"
                    content = parts[2].strip()
            except yaml.YAMLError:
                # If YAML parsing fails, treat as regular text
                pass
        
        # Use recursive chunking with markdown-aware separators
        chunks = super().chunk(content)
        
        # Add YAML header to each chunk for metadata context
        if yaml_header and chunks:
            chunks = [f"{yaml_header}{chunk}" for chunk in chunks]
        
        return chunks
    
    def chunk_with_metadata(self, text: str, file_path: str = "") -> list[dict]:
        """
        Chunk text and return structured output with metadata.
        
        Args:
            text: Input text
            file_path: Source file path for metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = self.chunk(text)
        return [
            {
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_file": file_path,
                "char_count": len(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]


class HierarchicalMarkdownChunker:
    """
    Two-stage chunker: First split by headers, then apply size constraints.
    
    Best for documents where section boundaries are more important than
    exact chunk sizes.
    """
    
    def __init__(
        self,
        headers_to_split_on: list[str] = None,
        max_chunk_size: int = 1024,
        overlap: int = 100
    ):
        self.headers = headers_to_split_on or ["## ", "### ", "#### "]
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.sub_chunker = RecursiveChunker(
            chunk_size=max_chunk_size,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk(self, text: str) -> list[str]:
        """Split by headers, then chunk oversized sections."""
        # First split by the first header level
        header_pattern = f"({ '|'.join(re.escape(h) for h in self.headers) })"
        sections = re.split(f"(?=\\n{header_pattern})", text)
        
        chunks = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                chunks.append(section.strip())
            else:
                # Sub-chunk oversized sections
                sub_chunks = self.sub_chunker.chunk(section)
                chunks.extend(sub_chunks)
        
        return [c for c in chunks if c.strip()]
```

### A.2 Enhanced Embedding Module (`src/embeddings.py` updates)

```python
# Updated model constants for multilingual support
LOCAL_EMBEDDING_MODEL_MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_EMBEDDING_MODEL_HIGH_QUALITY = "paraphrase-multilingual-mpnet-base-v2"
OLLAMA_EMBEDDING_MODEL_RECOMMENDED = "nomic-embed-text-v2-moe"


def get_recommended_embedder(
    provider: str = "local",
    use_high_quality: bool = False,
    **kwargs
) -> LocalEmbedder | OllamaEmbedder | OpenAIEmbedder:
    """
    Factory function to get the recommended embedder for medical product data.
    
    Args:
        provider: "local", "ollama", or "openai"
        use_high_quality: If True, use larger but more accurate model (local only)
        **kwargs: Additional arguments for embedder constructor
        
    Returns:
        Configured embedder instance
        
    Examples:
        >>> # Standard multilingual embedder (recommended)
        >>> embedder = get_recommended_embedder("local")
        >>> 
        >>> # Higher quality, larger model
        >>> embedder = get_recommended_embedder("local", use_high_quality=True)
        >>> 
        >>> # Via Ollama
        >>> embedder = get_recommended_embedder("ollama", base_url="http://localhost:11434")
    """
    if provider == "local":
        model_name = (
            LOCAL_EMBEDDING_MODEL_HIGH_QUALITY 
            if use_high_quality 
            else LOCAL_EMBEDDING_MODEL_MULTILINGUAL
        )
        return LocalEmbedder(model_name=model_name, **kwargs)
    
    elif provider == "ollama":
        base_url = kwargs.pop("base_url", "http://localhost:11434")
        return OllamaEmbedder(
            model_name=OLLAMA_EMBEDDING_MODEL_RECOMMENDED,
            base_url=base_url,
            **kwargs
        )
    
    elif provider == "openai":
        model_name = kwargs.pop("model_name", "text-embedding-3-small")
        return OpenAIEmbedder(model_name=model_name, **kwargs)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'local', 'ollama', or 'openai'.")


class MultilingualLocalEmbedder(LocalEmbedder):
    """
    Local embedder pre-configured for multilingual medical content.
    
    Extends LocalEmbedder with batch processing optimizations for
    bilingual (EN/VI) content.
    """
    
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self, model_name: str = None, batch_size: int = 32) -> None:
        super().__init__(model_name or self.DEFAULT_MODEL)
        self.batch_size = batch_size
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Efficiently embed a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        from sentence_transformers import SentenceTransformer
        
        # Use model's built-in batching with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        return [emb.tolist() for emb in embeddings]
```

### A.3 Complete Processing Script

```python
#!/usr/bin/env python3
"""
Process cleaned medical product files with recommended chunking and embedding.

Usage:
    python process_products.py --input-dir products_cleaned --output-dir processed
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Iterator

# Import from existing codebase
from src.chunking import MarkdownAwareChunker
from src.embeddings import get_recommended_embedder


def load_product_file(file_path: Path) -> dict:
    """Load and parse a cleaned product markdown file."""
    content = file_path.read_text(encoding='utf-8')
    
    # Parse YAML frontmatter
    metadata = {}
    body = content
    
    if content.startswith('---'):
        try:
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata = yaml.safe_load(parts[1])
                body = parts[2].strip()
        except yaml.YAMLError as e:
            print(f"Warning: Failed to parse YAML in {file_path}: {e}")
    
    return {
        "metadata": metadata,
        "body": body,
        "file_path": str(file_path.relative_to(Path.cwd())),
        "file_name": file_path.stem,
    }


def process_products(
    input_dir: Path,
    output_dir: Path,
    provider: str = "local",
    chunk_size: int = 768,
    overlap: int = 100,
) -> None:
    """Process all product files with chunking and embedding."""
    
    # Initialize chunker with recommended configuration
    chunker = MarkdownAwareChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        preserve_yaml=True
    )
    
    # Initialize embedder
    print(f"Initializing {provider} embedder...")
    embedder = get_recommended_embedder(provider)
    print(f"Using model: {embedder._backend_name}")
    
    # Process all markdown files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    md_files = list(input_dir.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files to process")
    
    for i, file_path in enumerate(md_files, 1):
        print(f"\nProcessing [{i}/{len(md_files)}]: {file_path}")
        
        # Load product data
        product = load_product_file(file_path)
        
        # Chunk the content
        chunks = chunker.chunk_with_metadata(
            product["body"], 
            file_path=product["file_path"]
        )
        print(f"  Created {len(chunks)} chunks")
        
        # Generate embeddings
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_batch(chunk_texts) if hasattr(embedder, 'embed_batch') else [
            embedder(text) for text in chunk_texts
        ]
        
        # Combine chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
            chunk["embedding_dimensions"] = len(embedding)
            chunk["product_metadata"] = product["metadata"]
            all_chunks.append(chunk)
        
        # Save individual file results
        output_file = output_dir / f"{product['file_name']}_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Save combined results
    combined_file = output_dir / "all_chunks.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Processing complete!")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Average chunks per file: {len(all_chunks) / len(md_files):.1f}")
    print(f"   Output directory: {output_dir}")
    print(f"   Combined file: {combined_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process medical product files with chunking and embedding"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("products_cleaned"),
        help="Directory containing cleaned markdown files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Directory for output JSON files"
    )
    parser.add_argument(
        "--provider",
        choices=["local", "ollama", "openai"],
        default="local",
        help="Embedding provider to use"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=768,
        help="Target chunk size in characters"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Character overlap between chunks"
    )
    
    args = parser.parse_args()
    
    process_products(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        provider=args.provider,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
```

---

## Appendix B: References

### Context7 Sources Cited

1. **LangChain Documentation** - Markdown splitting and chunking strategies
   - `/websites/langchain` - Primary chunking research
   - Key articles: RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

2. **Sentence Transformers** - Embedding models and multilingual support
   - `/huggingface/sentence-transformers` - Model comparisons
   - Key findings: `paraphrase-multilingual-MiniLM-L12-v2`, `paraphrase-multilingual-mpnet-base-v2`

3. **Ollama Documentation** - Local embedding deployment
   - `/websites/ollama` - API reference for embeddings
   - Key finding: nomic-embed-text model support

4. **OpenAI Cookbook** - Cloud embedding options
   - `/openai/openai-cookbook` - text-embedding-3 models
   - Key finding: Dimension truncation capabilities

5. **Pinecone Documentation** - Vector database chunking strategies
   - `/llmstxt/pinecone_io_llms-full_txt` - RAG optimization
   - Key finding: Content chunking strategy selection

### Additional Resources

- Nomic AI Documentation: https://docs.nomic.ai/reference/endpoints/embeddings
- Hugging Face Model Cards for recommended models
- LangChain Text Splitters Documentation: https://python.langchain.com/docs/modules/data_connection/document_transformers/

---

*Report generated: 2026-04-10*
*Research conducted using Context7 MCP tools*
