# Toku - RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) assistant for customer support. The system answers customer questions accurately with inline citations and enforces company policies.

## Features

### Core Capabilities
- **Hybrid Search**: Combines semantic search (sentence transformers + FAISS) with BM25 for robust retrieval
- **Cross-Encoder Reranking**: Improves precision with neural reranking
- **LLM-Based Generation**: Natural language answers using OpenAI/Anthropic (with offline fallback)
- **Inline Citations**: Every fact is cited in format `[source_file#identifier]`
- **Confidence Scoring**: Quality assessment for each answer

### Policy Enforcement
- **PII Masking**: Automatically redacts card numbers, national IDs, masks phone numbers
- **Escalation Detection**: Identifies P0/P1/P2 severity levels and provides guidance
- **SLA Compliance**: Differentiates enterprise (4hr) vs standard (24hr) response times

## Project Structure

```
.
├── config.py              # Configuration management
├── data_loader.py         # Data loading with intelligent chunking
├── retriever.py           # Hybrid search + reranking
├── generator.py           # LLM-based answer generation
├── policy_enforcer.py     # Policy enforcement
├── assistant.py           # Main orchestrator
├── evaluate.py            # Evaluation script
├── main.py                # Entry point (interactive/eval modes)
├── indexer.py             # Legacy indexer (backward compatibility)
├── data/
│   ├── plans.csv          # Telecom plans
│   ├── kb.yaml            # Knowledge base (policies, SLAs, discounts)
│   ├── transcripts.json   # Customer support examples
│   ├── faq.jsonl          # FAQ pairs
│   └── eval_prompts.txt   # Evaluation questions
├── requirements.txt       # Python dependencies
├── ARCHITECTURE.md        # Design documentation
└── README.md              # This file
```

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up LLM API key for enhanced generation:
```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Or for Anthropic
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Note**: The system works fully without an API key using offline template-based generation.

### First Run

The first run will download the embedding models (~90MB), which may take a few minutes.

## Usage

### Run Evaluation

Evaluate the system on `eval_prompts.txt`:

```bash
python evaluate.py
```

This will:
- Load all data sources
- Build hybrid search index
- Run evaluation on all prompts
- Save results to `evaluation_outputs.json` and `evaluation_outputs.txt`

### Interactive Mode

Run the assistant interactively:

```bash
python main.py
```

Commands in interactive mode:
- Type customer questions to get answers
- `help` - Show available commands
- `eval` - Run evaluation
- `explain` - Toggle explanation mode (shows retrieval details)
- `status` - Show assistant status
- `clear` - Clear screen
- `exit` - Quit

### Command Line Options

```bash
# Run in offline mode (no LLM API needed)
python main.py --offline

# Disable reranking (faster but less precise)
python main.py --no-rerank

# Run evaluation directly
python main.py --eval
```

### Programmatic Usage

```python
from assistant import RAGAssistant, create_assistant

# Create and initialize assistant
assistant = create_assistant(use_llm=True, use_reranker=True)
assistant.initialize()

# Generate answer
response = assistant.generate_answer(
    "Which plan includes call recording?",
    top_k=5,
    is_enterprise=False
)

print(response["answer"])
print("Citations:", response["citations"])
print("Confidence:", f"{response['confidence']:.2%}")

# Get detailed explanation
explanation = assistant.explain_answer("Customer wants SSO support")
print(explanation["breakdown"]["retrieved_documents"])
```

## Output Format

### Evaluation Outputs

Evaluation produces two files:

1. **evaluation_outputs.json**: Structured JSON with all results
2. **evaluation_outputs.txt**: Human-readable text format

Each result includes:
- Query and answer with inline citations
- List of all citations used
- Confidence score
- Escalation information (if triggered)
- SLA guidance
- Generation method used

### Citation Format

Citations follow the format: `[source_file#identifier]`

Examples:
- `[plans.csv#row=4]` - Row 4 in plans.csv
- `[kb.yaml#features_matrix]` - Features matrix section
- `[kb.yaml#discounts]` - Discount rules section
- `[transcripts.json#t-002]` - Transcript t-002
- `[faq.jsonl#entry=2]` - FAQ entry 2

## Configuration

Configuration can be set via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `LLM_PROVIDER` | "openai", "anthropic", or "offline" | "openai" |
| `LLM_MODEL` | Model name | "gpt-4o-mini" |
| `USE_RERANKER` | Enable cross-encoder reranking | "true" |
| `SEMANTIC_WEIGHT` | Weight for semantic search (0-1) | 0.6 |
| `BM25_WEIGHT` | Weight for BM25 search (0-1) | 0.4 |

## How It Works

### Pipeline Overview

1. **Data Loading**: All sources loaded into `Document` objects with metadata and keywords
2. **Indexing**: 
   - Semantic: Sentence transformer embeddings → FAISS index
   - BM25: Custom tokenization → inverted index
3. **Query Processing**:
   - Query expansion with domain synonyms
   - Parallel hybrid search (semantic + BM25)
   - Score fusion with configurable weights
   - Cross-encoder reranking for top results
4. **Answer Generation**:
   - LLM mode: GPT-4o-mini with grounding constraints
   - Offline mode: Template-based with intent detection
5. **Policy Enforcement**: PII masking, escalation detection, SLA guidance
6. **Response**: Answer with inline citations, confidence score, policy info

### Key Algorithms

- **Hybrid Search**: `final_score = 0.7 * semantic_score + 0.3 * bm25_score`
- **Hybrid Search**: `final_score = 0.6 * semantic_score + 0.4 * bm25_score`
- **Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2) for precision
- **Query Expansion**: Domain-specific synonyms (e.g., "SSO" → "single sign-on")

## Design Decisions

See `ARCHITECTURE.md` for detailed design decisions, trade-offs, and improvement roadmap.

Key highlights:
- **Hybrid search** provides robustness against both semantic and keyword misses
- **Offline mode** ensures reproducibility and works without API dependencies
- **Inline citations** enable precise fact traceability
- **Modular architecture** allows easy extension and testing

## Evaluation Criteria

The system is evaluated on:
- **Grounding**: Answers traceable to the dataset
- **Citations**: Proper format and accuracy
- **Policy Enforcement**: PII masking, escalation, SLA compliance
- **Answer Quality**: Relevance, completeness, natural language
- **Confidence**: Score reflects retrieval quality

## Troubleshooting

### Model Download Issues
If the sentence transformer model fails to download:
- Check internet connection (first run only)
- Models are cached locally after first download
- Try: `pip install --upgrade sentence-transformers`

### Memory Issues
If you encounter memory issues:
- Reduce `top_k` parameter in `generate_answer()`
- Disable reranking with `--no-rerank`
- Use a smaller embedding model

### LLM API Errors
If LLM generation fails:
- Check API key is set correctly
- Verify API quota/billing
- System automatically falls back to offline mode

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Performance

| Metric | Value |
|--------|-------|
| Index Build | ~3-5 seconds |
| Query (offline) | ~200ms |
| Query (LLM) | ~1-2 seconds |
| Memory | ~200MB |
