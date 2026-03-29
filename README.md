# CommerceIQ -  AI Data Analyst

A local-first AI data copilot for exploring datasets with natural language.

This application lets users upload or connect datasets, ask questions in plain English, generate insights, build visualizations, and even create SQL queries automatically. It is designed specifically for dataset exploration and analytics, not as a general-purpose coding assistant.

Unlike cloud-only AI tools, this system can run LLMs locally, helping teams keep sensitive data inside their own environment for better privacy, compliance, and control.

Features

Upload and analyze CSV, XLSX, and JSON files

Ask natural language questions about your data

View extracted dataset metadata such as:
- column names
- data types
- schema details

Generate:
- summaries and insights
- charts and visualizations

SQL queries

- Work with multiple datasets at once
- Support automatic joins when data modeling templates are provided
- Perform aggregations across related datasets
- Run with local LLMs and vector DBs for privacy-first analytics

Connect to cloud storage sources including:
- Azure Data Lake Storage (ADLS)
- Amazon S3
- other cloud-integrated data sources

## Technology Stack

Programming & Data Processing
- Python
- Pandas
- NumPy

Visualization
- Plotly
- Matplotlib

AI & LLM Integration
- Ollama (llama3.2)
- LangChain

Vector Database
- ChromaDB

User Interface
- Gradio

## Setup

1. **Create and use the project virtual environment** (recommended — venv lives inside `ai_data_analyst`):
   ```bash
   cd ai_data_analyst
   python3 -m venv myenv
   source myenv/bin/activate   # Windows: myenv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Make sure you have [Ollama](https://ollama.com/) installed and running locally with the `llama3.2` model available:
   ```bash
   ollama pull llama3.2
   ```

3. Run the application:
   ```bash
   python app.py
   ```
   If you use the project venv, run from the same directory so `./data` and imports resolve correctly.

## Embedding Model Configuration

The app uses an open-source Hugging Face embedding model for RAG.

- Default (higher quality, free): `BAAI/bge-base-en-v1.5`
- Faster/lighter option: `sentence-transformers/all-MiniLM-L6-v2`
- Higher quality/heavier option: `BAAI/bge-large-en-v1.5`

Configure with environment variables:

```bash
export EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
export EMBEDDING_DEVICE="cpu"            # or "cuda" if available
export EMBEDDING_NORMALIZE="true"        # recommended
export EMBEDDING_BATCH_SIZE="32"         # optional
```

Important: if you change `EMBEDDING_MODEL`, rebuild/reindex your vector DB (`data/chroma_db`) so indexed vectors and query vectors stay in the same embedding space.
