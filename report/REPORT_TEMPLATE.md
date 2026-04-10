# BÁO CÁO PHA 2: RAG Implementation & Evaluation

**Thành viên:** [Tên sinh viên]  
**MSSV:** [Mã số sinh viên]  
**Ngày:** 10/04/2026  
**Pha:** 2 - RAG Implementation & Evaluation  
**Chế độ:** Option A (Safe Mode)

---

## 1. TÓM TẮT

Báo cáo này trình bày việc triển khai và đánh giá hệ thống RAG cho tài liệu sản phẩm y tế (medical devices từ 3 hãng: Schwind, Melag, BVI). Hệ thống sử dụng chiến lược **RecursiveChunker** với các separators markdown-aware.

**Thông số kỹ thuật:**
- **Chunker:** RecursiveChunker với separators: `["\n## ", "\n### ", "\n\n", "\n", ". ", " " ]`
- **Chunk Size:** 768 characters
- **Embedding Model:** nomic-embed-text-v2-moe (OllamaEmbedder)
- **Vector Store:** EmbeddingStore với metadata support
- **Dữ liệu:** 110 files (55 EN + 55 VI) từ 3 brands

---

## 2. KẾT QUẢ ĐÁNH GIÁ

### 2.1 Retrieval Precision (Top-k relevance)

| Query ID | Query | Precision | Keywords Found | Expected Brand |
|---------|-------|-----------|----------------|----------------|
| 1 | laser eye surgery technology | [X%] | [keywords] | schwind |
| 2 | medical sterilization equipment | [X%] | [keywords] | melag |
| 3 | phaco equipment ophthalmic surgery | [X%] | [keywords] | bvi |
| 4 | Bowie Dick sterilization control | [X%] | [keywords] | melag |
| 5 | cornea treatment therapeutic solutions | [X%] | [keywords] | schwind |

**Phân tích:** [Mô tả ngắn về độ chính xác của retrieval]

### 2.2 Chunk Coherence

| Query ID | Coherence Score | Nhận xét |
|---------|-----------------|----------|
| 1 | [X%] | [Nhận xét] |
| 2 | [X%] | [Nhận xét] |
| 3 | [X%] | [Nhận xét] |
| 4 | [X%] | [Nhận xét] |
| 5 | [X%] | [Nhận xét] |

**Trung bình:** [X%]

**Phân tích:** [Đánh giá việc chunk có giữ được ý nghĩa trọn vẹn không]

### 2.3 Metadata Filter Utility

| Query ID | Unfiltered Precision | Filtered Precision | Improvement |
|---------|---------------------|-------------------|-------------|
| 1 | [X%] | [X%] | [+X%] |
| 2 | [X%] | [X%] | [+X%] |
| 3 | [X%] | [X%] | [+X%] |
| 4 | [X%] | [X%] | [+X%] |
| 5 | [X%] | [X%] | [+X%] |

**Phân tích:** [Đánh giá tác dụng của metadata filtering]

### 2.4 Grounding Quality

| Query ID | Grounding Score | Content Length | Metadata Completeness |
|---------|-----------------|----------------|----------------------|
| 1 | [X%] | [chars] | [X%] |
| 2 | [X%] | [chars] | [X%] |
| 3 | [X%] | [chars] | [X%] |
| 4 | [X%] | [chars] | [X%] |
| 5 | [X%] | [chars] | [X%] |

**Trung bình:** [X%]

**Phân tích:** [Đánh giá khả năng grounding cho agent]

### 2.5 Data Strategy Impact

**Tổng số documents:** [X]

**Brand Coverage:**
| Brand | Documents | Tỷ lệ |
|-------|-----------|-------|
| schwind | [X] | [X%] |
| melag | [X] | [X%] |
| bvi | [X] | [X%] |

**Brand Balance Score:** [X%]

**Category Coverage:**
| Category | Results |
|----------|---------|
| technology | [X] |
| equipment | [X] |
| devices | [X] |
| control_systems | [X] |
| therapy | [X] |

**Phân tích:** [Đánh giá sự phù hợp của bộ tài liệu với benchmark queries]

---

## 3. PHÂN TÍCH CHI TIẾT

### 3.1 Ví dụ Top-k Results

**Query 1: "laser eye surgery technology"**

| Rank | Score | Brand | Product | Language | Relevance |
|------|-------|-------|---------|----------|-----------|
| 1 | [X.XXXX] | [brand] | [product] | [lang] | [✓/✗] |
| 2 | [X.XXXX] | [brand] | [product] | [lang] | [✓/✗] |
| 3 | [X.XXXX] | [brand] | [product] | [lang] | [✓/✗] |

**Nhận xét:** [Phân tích chi tiết về kết quả]

### 3.2 So sánh Filter vs No-Filter

**Query:** [Ví dụ]

**Không filter (top-3):**
1. [Brand A] - [Product] - Score: [X.XXXX]
2. [Brand B] - [Product] - Score: [X.XXXX]
3. [Brand C] - [Product] - Score: [X.XXXX]

**Có filter (brand = [expected]):**
1. [Brand X] - [Product] - Score: [X.XXXX]
2. [Brand X] - [Product] - Score: [X.XXXX]
3. [Brand X] - [Product] - Score: [X.XXXX]

**Kết luận:** [Nhận xét về hiệu quả filtering]

---

## 4. ĐÁNH GIÁ KẾT QUẢ NHÓM (GROUP)

*[Phần này tổng hợp kết quả của cả nhóm - điền sau khi thảo luận nhóm]*

### 4.1 So sánh các chiến lược trong nhóm

| Thành viên | Chunker Strategy | Avg Precision | Avg Coherence | Filter Impact |
|-----------|------------------|---------------|---------------|---------------|
| [Tên] | RecursiveChunker | [X%] | [X%] | [+X%] |
| [Tên] | [Strategy] | [X%] | [X%] | [+X%] |
| [Tên] | [Strategy] | [X%] | [X%] | [+X%] |

### 4.2 Nhận xét chung

**Điểm mạnh:**
- [Điểm mạnh 1]
- [Điểm mạnh 2]

**Điểm yếu:**
- [Điểm yếu 1]
- [Điểm yếu 2]

**Bài học:**
- [Bài học 1]
- [Bài học 2]

---

## 5. KẾT LUẬN VÀ KHUYẾN NGHỊ

### 5.1 Tổng kết

*[Tóm tắt ngắn gọn kết quả đạt được]*

### 5.2 Khuyến nghị cải thiện

1. **[Khuyến nghị 1]** - [Giải thích]
2. **[Khuyến nghị 2]** - [Giải thích]
3. **[Khuyến nghị 3]** - [Giải thích]

### 5.3 Hướng phát triển

*[Gợi ý cho các pha tiếp theo]*

---

## 6. PHỤ LỤC

### 6.1 Câu lệnh chạy scripts

```bash
# Xử lý dữ liệu
python process_products.py

# Chạy đánh giá
python evaluate_strategy.py

# Chạy tests
pytest tests/ -v
```

### 6.2 File artifacts

| File | Mô tả |
|------|-------|
| `process_products.py` | Script xử lý 110 files, tạo embeddings |
| `benchmark_queries.py` | 5 benchmark queries với expected results |
| `evaluate_strategy.py` | Script đánh giá theo 5 tiêu chí |
| `report/REPORT.md` | Báo cáo kết quả điền đầy đủ |

### 6.3 Thống kê processing

- **Total files processed:** 110
- **Total chunks:** [X]
- **Average chunks per file:** [X]
- **Processing time:** [X seconds]

---

## 7. REFERENCES

- [1] Project README: `/home/thanhnndev/develop/ai.20k/2A202600250-NongNguyenThanh-Day-07/README.md`
- [2] Cleaned data: `/home/thanhnndev/develop/ai.20k/2A202600250-NongNguyenThanh-Day-07/products_cleaned/`
- [3] Source code: `/home/thanhnndev/develop/ai.20k/2A202600250-NongNguyenThanh-Day-07/src/`

---

**Xác nhận hoàn thành:**
- [x] Task 1: Processing script
- [x] Task 2: Benchmark queries
- [x] Task 3: Evaluation script
- [x] Task 4: Report template
- [ ] Report filled with actual results (sau khi chạy evaluation)
- [ ] Git commits hoàn tất

---

*Điền template này với kết quả thực tế sau khi chạy `python evaluate_strategy.py`*
