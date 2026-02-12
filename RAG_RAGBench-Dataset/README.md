## RAGBench Dataset – Overview

RAGBench is a benchmark dataset designed to evaluate Retrieval-Augmented Generation (RAG) systems. It measures how effectively a model retrieves relevant context and generates accurate responses.

### Key Components:
- **id**: Unique identifier for each sample.
- **question**: The input query posed to the system.
- **documents**: Retrieved documents associated with the query.
- **documents_sentences**: Sentence-level breakdown of the retrieved documents.
- **response**: Ground-truth answer.
- **all_relevant_sentence_keys**: Identifiers for sentences considered relevant to answering the question.

### Purpose:
RAGBench evaluates both:
- **Retrieval quality** (Are relevant sentences retrieved?)
- **Generation quality** (Is the final answer accurate and grounded?)

It supports multiple domains such as:
- `covidqa`
- `hotpotqa`
- `finqa`
- `cuad`
- `msmarco`
- and others.
