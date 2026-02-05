# TokuTel RAG Assistant

A retrieval-augmented generation (RAG) assistant for TokuTel customer support that answers questions accurately with citations and enforces company policies.

## Features

- **Semantic Search**: Uses sentence transformers and FAISS for efficient document retrieval
- **Citation Tracking**: Every answer includes citations in format `[source_file#identifier]`
- **Policy Enforcement**: 
  - PII masking (card numbers, national IDs, phone numbers)
  - Escalation detection (P0/P1/P2)
  - SLA guidance (enterprise vs standard)
- **Evaluation**: Automated evaluation on provided prompts

## Project Structure

```
.
├── data/
│   ├── plans.csv          # Telecom plans data
│   ├── kb.yaml            # Knowledge base (policies, SLAs, discounts)
│   ├── transcripts.json   # Customer support examples
│   ├── faq.jsonl          # FAQ pairs
│   └── eval_prompts.txt   # Evaluation questions
├── data_loader.py         # Data loading and parsing
├── indexer.py             # Vector indexing with embeddings
├── policy_enforcer.py     # Policy enforcement logic
├── assistant.py           # Main RAG assistant
├── evaluate.py            # Evaluation script
├── main.py                # Entry point
├── requirements.txt       # Python dependencies
├── ARCHITECTURE.md        # Design documentation
└── README.md              # This file
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The first run will download the sentence transformer model (~90MB), which may take a few minutes.

## Usage

### Run Evaluation

Evaluate the system on `eval_prompts.txt`:

```bash
python evaluate.py
```

This will:
- Load all data sources
- Create vector index
- Run evaluation on all prompts
- Save results to `evaluation_outputs.json` and `evaluation_outputs.txt`

### Interactive Mode

Run the assistant interactively:

```bash
python main.py
```

In interactive mode:
- Type customer questions to get answers
- Type `eval` to run evaluation
- Type `exit` to quit

### Programmatic Usage

```python
from assistant import TokuTelAssistant

assistant = TokuTelAssistant()
assistant.initialize()

response = assistant.generate_answer(
    "Which plan includes call recording?",
    top_k=5,
    is_enterprise=False
)

print(response["answer"])
print("Citations:", response["citations"])
```

## Output Format

### Evaluation Outputs

Evaluation produces two files:

1. **evaluation_outputs.json**: Structured JSON with all results
2. **evaluation_outputs.txt**: Human-readable text format

Each result includes:
- Query
- Answer with citations
- Escalation information (if needed)
- SLA guidance

### Citation Format

Citations follow the format: `[source_file#identifier]`

Examples:
- `[plans.csv#row=4]` - Row 4 in plans.csv
- `[kb.yaml#features_matrix]` - Features matrix section in kb.yaml
- `[transcripts.json#t-002]` - Transcript t-002
- `[faq.jsonl#entry=2]` - Entry 2 in faq.jsonl

## How It Works

1. **Data Loading**: All data sources are loaded and structured into documents with citations
2. **Indexing**: Documents are embedded using sentence transformers and indexed in FAISS
3. **Retrieval**: User queries are embedded and matched against the index
4. **Answer Generation**: Answers are constructed from retrieved documents
5. **Policy Enforcement**: PII masking, escalation detection, and SLA guidance are applied
6. **Response**: Final answer with citations and policy information is returned

## Design Decisions

See `ARCHITECTURE.md` for detailed design decisions, trade-offs, and potential improvements.

## Evaluation Criteria

The system is evaluated on:
- **Grounding**: Answers must be traceable to the dataset
- **Citations**: Proper citation format and accuracy
- **Policy Enforcement**: PII masking, escalation detection, SLA compliance
- **Answer Quality**: Relevance and completeness

## Limitations

- Answer generation uses template-based approach (not LLM-based)
- Escalation detection is keyword-based
- No multi-turn conversation support
- Fixed retrieval count (k=5)

See `ARCHITECTURE.md` for detailed limitations and improvement suggestions.

## Troubleshooting

### Model Download Issues

If the sentence transformer model fails to download:
- Check internet connection (first run only)
- The model will be cached locally after first download

### Memory Issues

If you encounter memory issues:
- Reduce `top_k` parameter in `generate_answer()`
- Use a smaller embedding model (modify `model_name` in `Indexer`)

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## License

This is a take-home assignment for TokuTel AI Engineer position.

## Contact

For questions about this assignment, contact Haniyeh at Haniyeh.abdi@toku.co
