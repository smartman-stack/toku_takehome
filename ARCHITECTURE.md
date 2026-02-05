# TokuTel RAG Assistant - Architecture Document

## Overview

This document describes the design, implementation, and trade-offs of the TokuTel Retrieval-Augmented Generation (RAG) assistant system. The system is designed to answer customer questions accurately while enforcing company policies and providing traceable citations.

## System Architecture

### Components

1. **Data Loader** (`data_loader.py`)
   - Loads and parses all data sources (CSV, YAML, JSON, JSONL)
   - Structures data into `Document` objects with citation metadata
   - Ensures each document chunk has a traceable citation identifier

2. **Indexer** (`indexer.py`)
   - Creates vector embeddings using sentence transformers (all-MiniLM-L6-v2)
   - Builds FAISS index for efficient similarity search
   - Uses cosine similarity for document retrieval

3. **Retriever** (integrated in `assistant.py`)
   - Performs semantic search over indexed documents
   - Returns top-k most relevant documents with similarity scores
   - Maintains citation information throughout retrieval

4. **Policy Enforcer** (`policy_enforcer.py`)
   - Masks PII (card numbers, national IDs, phone numbers)
   - Detects escalation needs based on query content
   - Applies SLA guidance based on customer type

5. **Assistant** (`assistant.py`)
   - Orchestrates retrieval, answer generation, and policy enforcement
   - Constructs answers from retrieved context
   - Formats responses with citations

6. **Evaluator** (`evaluate.py`)
   - Runs evaluation on provided prompts
   - Saves structured outputs (JSON and human-readable)

## Design Decisions

### 1. Embedding Model Choice

**Decision**: Use `all-MiniLM-L6-v2` sentence transformer model

**Rationale**:
- Works offline without API keys (reproducibility)
- Lightweight (384 dimensions) and fast
- Good balance between quality and speed
- Sufficient for this dataset size

**Trade-off**: Larger models (e.g., OpenAI embeddings) might provide better semantic understanding but require API access and add latency/cost.

### 2. Indexing Strategy

**Decision**: Create separate documents for each logical unit (plan row, policy section, transcript, FAQ entry)

**Rationale**:
- Enables precise citation tracking
- Maintains context boundaries
- Allows retrieval of specific policy sections or plan details

**Trade-off**: Could chunk larger documents further for better granularity, but current approach balances specificity with context preservation.

### 3. Retrieval Approach

**Decision**: Use semantic search with top-k retrieval (k=5)

**Rationale**:
- Captures semantic similarity beyond keyword matching
- Multiple documents provide comprehensive context
- Top-k allows ranking by relevance

**Trade-off**: Could use hybrid search (semantic + keyword) for better recall, but semantic search alone is sufficient for this use case.

### 4. Answer Generation

**Decision**: Template-based answer construction from retrieved documents

**Rationale**:
- No external LLM API required (reproducibility, cost)
- Deterministic outputs for evaluation
- Direct grounding in retrieved documents

**Trade-off**: 
- **Current**: Simple, deterministic, fully grounded
- **Alternative**: LLM-based generation would provide more natural language but requires API access, adds complexity, and risks hallucination

**Future Improvement**: In production, use a local LLM (e.g., Llama 2) or API-based LLM with strict grounding constraints.

### 5. Citation Format

**Decision**: Use format `[source_file#identifier]` (e.g., `[plans.csv#row=4]`)

**Rationale**:
- Clear and parseable
- Points directly to source location
- Consistent across all data sources

### 6. Policy Enforcement

**Decision**: Apply policies post-generation

**Rationale**:
- Ensures all outputs comply regardless of generation method
- Centralized policy logic
- Easy to audit and modify

**Implementation Details**:
- **PII Masking**: Regex-based detection and masking
- **Escalation**: Keyword-based detection (could be improved with classification)
- **SLA**: Simple flag-based (enterprise vs standard)

## Data Flow

```
User Query
    ↓
Assistant.generate_answer()
    ↓
Indexer.search() → Retrieve top-k documents
    ↓
Construct answer from retrieved context
    ↓
PolicyEnforcer.enforce_policies()
    ├─ Mask PII
    ├─ Check escalation needs
    └─ Apply SLA guidance
    ↓
Return response with citations
```

## Limitations and Future Improvements

### Current Limitations

1. **Answer Generation**: Template-based approach is limited in natural language quality
2. **Escalation Detection**: Keyword-based, could miss nuanced cases
3. **No Multi-turn Context**: Each query is independent
4. **Fixed Retrieval**: Always retrieves k=5 documents regardless of query complexity

### Potential Improvements

1. **Hybrid Search**: Combine semantic search with BM25 keyword matching
2. **Reranking**: Use cross-encoder for better relevance ranking
3. **LLM Integration**: Use local LLM (Llama 2, Mistral) for natural answer generation
4. **Query Understanding**: Classify query intent (plan inquiry, feature request, escalation)
5. **Contextual Retrieval**: Use query expansion or reformulation
6. **Multi-turn Support**: Maintain conversation context across turns
7. **Confidence Scoring**: Provide confidence scores for answers
8. **Better Escalation**: Train classifier for escalation detection
9. **Dynamic Retrieval**: Adjust k based on query complexity or confidence
10. **Caching**: Cache common queries for faster responses

## Production Readiness Considerations

### What's Production-Ready

- ✅ Citation tracking and traceability
- ✅ Policy enforcement (PII masking, escalation, SLA)
- ✅ Reproducible and deterministic
- ✅ Works offline
- ✅ Clear code structure

### What Needs Work for Production

- ⚠️ Answer generation quality (needs LLM)
- ⚠️ Error handling and edge cases
- ⚠️ Performance optimization (caching, async)
- ⚠️ Monitoring and logging
- ⚠️ Testing suite
- ⚠️ Configuration management
- ⚠️ Multi-language support (if needed)

## Evaluation Approach

The system is evaluated on `eval_prompts.txt` with:
- Answer quality and grounding
- Citation accuracy
- Policy enforcement (PII masking, escalation detection)
- SLA guidance application

Outputs are saved in both JSON (structured) and text (human-readable) formats for review.

## Conclusion

This system provides a solid foundation for a RAG-based customer support assistant with strong grounding, citation tracking, and policy enforcement. The design prioritizes reproducibility and traceability while leaving room for enhancement with LLM-based generation and more sophisticated retrieval strategies in production.
