# RapidFire-AI-Winter-LLM-Challenge
---
## RAG CHALLENGE 
We used 2 different datasets for this Challenge - "DOCUGAMI Financial Dataset" and "RAGBench Dataset (CovidQA)".

Below we discuss the comparison of performances between the  datasets, and the possible ways/causes/reasons for different resulting metrics for each of the datasets. 
# CHUNKING COMPARISON: DOCUGAMI Financial Dataset vs RAGBench

## DOCUGAMI FINANCIAL DATASET - The Problem

### Step 1: Corpus Creation (in your Python script)
```
PDF File: "2023_Q3_AAPL.pdf" (50,000 characters)
                    ↓
        extract_pdf_text()
                    ↓
    RecursiveCharacterTextSplitter
    (chunk_size=512, chunk_overlap=128)
                    ↓
┌─────────────────────────────────────────┐
│ doc_0: "Apple Inc. quarterly..."        │ 512 chars
│ doc_1: "...revenue increased by..."     │ 512 chars  
│ doc_2: "...operating expenses were..."  │ 512 chars
│ ... (continuing)                        │
└─────────────────────────────────────────┘
        Saved to: corpus_sampled.jsonl
```

### Step 2: RAG Loading (WRONG - causes double chunking)
```
Load corpus_sampled.jsonl
                    ↓
┌─────────────────────────────────────────┐
│ doc_0: "Apple Inc. quarterly..."        │ 512 chars
│ corpus_id = "doc_0"                     │
└─────────────────────────────────────────┘
                    ↓
    RecursiveCharacterTextSplitter  ← PROBLEM!
    (chunk_size=512, chunk_overlap=128)
                    ↓
┌─────────────────────────────────────────┐
│ NEW_CHUNK_0: "Apple Inc. qua..."        │ 512 chars
│ corpus_id = "doc_0" (duplicated!)       │
│                                         │
│ NEW_CHUNK_1: "...quarterly rep..."      │ 384 chars (overlap region)
│ corpus_id = "doc_0" (duplicated!)       │
└─────────────────────────────────────────┘

RESULT: Chunk boundaries change!
        Metadata gets duplicated!
        Retrieved corpus_ids don't match QRELS!
```

### Step 3: Evaluation Failure
```
QRELS says:           RAG retrieves:
query_id: q_0         query_id: q_0
corpus_id: doc_5      corpus_id: doc_5_0  ← MISMATCH!
relevance: 1          corpus_id: doc_5_1  ← MISMATCH!
                      corpus_id: doc_6_0

Precision = 0 / 3 = 0%  ← Nothing matches!
Recall = 0 / 1 = 0%
```

### THE FIX
```
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=100000,  ← Much larger than your 512-char chunks
    chunk_overlap=0,
)

Since 512 < 100000, no splitting occurs!
Chunks remain: doc_0, doc_1, doc_2... (exactly as in QRELS)
```

---

## RAGBENCH DATASET - How It Works

### Data Structure (Already Pre-Chunked)
```
Sample 358 from RAGBench CovidQA:
{
  "id": "358",
  "question": "What role does T-cell count play in...",
  
  "documents": [
    "Title: Emergent severe acute respiratory...",  ← Full document 0
    "Title: Emergent severe acute respiratory...",  ← Full document 1
    "Title: Human adenovirus type 7...",            ← Full document 2
    "Title: Human adenovirus type 7..."             ← Full document 3
  ],
  
  "documents_sentences": [
    [  ← Document 0 sentences
      "Title: Emergent severe acute...",           ← Sentence 0a
      "Passage: Recent studies have...",           ← Sentence 0b
      "Chen et al. reported that...",              ← Sentence 0c
      "In our study, the only patient...",         ← Sentence 0d
      "Three of the five patients...",             ← Sentence 0e
      "Our results suggest that..."                ← Sentence 0f
    ],
    [  ← Document 1 sentences
      "Title: Emergent severe...",                 ← Sentence 1a
      "Passage: Recent studies...",                ← Sentence 1b
      ...
    ],
    ...
  ],
  
  "all_relevant_sentence_keys": ["0d", "0e", "0f", "1d", "1e", "1f"]
}
```

### Corpus Creation (Sentence-Level Chunks)
```
For each sample:
  For each document in sample:
    For each sentence in document:
      Create chunk:
        corpus_id = "358_doc0_sent0"
        sent_key = "0a"
        text = "Title: Emergent severe acute..."
        
Result:
┌──────────────────────────────────────────────────────┐
│ corpus_id: "358_doc0_sent0"                          │
│ sent_key: "0a"                                       │
│ text: "Title: Emergent severe acute..."             │
├──────────────────────────────────────────────────────┤
│ corpus_id: "358_doc0_sent1"                          │
│ sent_key: "0b"                                       │
│ text: "Passage: Recent studies have..."             │
├──────────────────────────────────────────────────────┤
│ corpus_id: "358_doc0_sent3"                          │
│ sent_key: "0d"  ← RELEVANT (in all_relevant_keys)   │
│ text: "In our study, the only patient..."           │
└──────────────────────────────────────────────────────┘
```

### QRELS Creation
```
For query "358":
  Relevant chunks (relevance=1):
    - 358_doc0_sent3 (key "0d")
    - 358_doc0_sent4 (key "0e")
    - 358_doc0_sent5 (key "0f")
    - 358_doc1_sent3 (key "1d")
    - 358_doc1_sent4 (key "1e")
    - 358_doc1_sent5 (key "1f")
  
  Irrelevant chunks (relevance=0):
    - All other sentences
```

### RAG Loading (NO Re-chunking)
```
Load sentences as-is:
┌──────────────────────────────────────────┐
│ "In our study, the only patient..."      │ ~60 chars
│ corpus_id = "358_doc0_sent3"             │
└──────────────────────────────────────────┘
                    ↓
    RecursiveCharacterTextSplitter
    (chunk_size=100000, chunk_overlap=0)  ← Large size!
                    ↓
┌──────────────────────────────────────────┐
│ "In our study, the only patient..."      │ Still ~60 chars
│ corpus_id = "358_doc0_sent3"             │ ← PRESERVED!
└──────────────────────────────────────────┘

No splitting occurs! Corpus IDs match QRELS perfectly!
```

---

## KEY DIFFERENCES SUMMARY

| Aspect | Your Financial Dataset | RAGBench |
|--------|------------------------|----------|
| **Original Format** | PDF files | Text passages in JSON |
| **Pre-processing** | Extract text, chunk at 512 chars | Already sentence-chunked |
| **Chunk Size** | ~512 characters | ~50-150 characters (sentence length) |
| **Chunk Count** | ~100 chunks per 50KB PDF | ~20-30 sentences per document |
| **Ground Truth** | Inferred from answer keywords | Human-annotated sentence keys |
| **Corpus IDs** | `doc_0`, `doc_1`, `doc_2`... | `sample_id_doc0_sent0`, or `"0a"`, `"0b"`... |
| **Re-chunking Issue** | ✅ YES - broke alignment | ✅ YES - would break alignment |
| **Solution** | `chunk_size=100000` (no split) | `chunk_size=100000` (no split) |

---

## PRACTICAL EXAMPLES

### Example 1: DOCUGAMI Financial Dataset
```python
# BEFORE (WRONG - 7% precision)
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=512,      # Same as corpus chunks
    chunk_overlap=128    # Creates new boundaries!
)
# → Retrieved: doc_0_0, doc_0_1, doc_1_0
# → QRELS expects: doc_0, doc_1
# → Mismatch = 0% precision
```

### Example 2: RAGBench
```python
# Corpus has sentence-level chunks (~60 chars each)
# Don't re-chunk them!

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=100000,   # Prevents splitting sentences
    chunk_overlap=0
)

# Retrieved corpus_ids: ["358_doc0_sent3", "358_doc0_sent4"]
# QRELS relevant: ["358_doc0_sent3", "358_doc0_sent4", "358_doc0_sent5"]
# Match: 2 out of 2 retrieved are relevant
# Precision: 100%, Recall: 67%
```

---

The precision observed for RAG Bench is currently low. This can primarily be attributed to the following factors:

On average, each query in the dataset has approximately 20 relevant sentences. However, in our setup, we restrict retrieval to only the top 5 sentences. As a result, the precision appears low because many relevant sentences are not included in the evaluated subset. Attempting to increase the retrieval size to the top 20 sentences using the cross-reranker led to GPU memory and computation constraints.

The same limitation contributes to the relatively low performance observed across other evaluation metrics as well.

Given the fixed architectural constraints of the RapidFire API calls and the existing GPU limitations, significant modifications to the retrieval pipeline are not feasible. Therefore, using a dataset with a smaller number of relevant documents per query would be a more suitable approach under the current constraints.


