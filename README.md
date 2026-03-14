# AI Data Analyst

A local AI data copilot that lets users explore datasets using natural language. Users can upload CSV,XLSX or JSON files and ask questions to get insights, generate plots, or even create SQL queries automatically.

Unlike general AI tools like ChatGPT or GitHub Copilot, this application is designed specifically for dataset exploration and analytics. It allows users to upload multiple local datasets, view extracted metadata such as columns and data types, and ask natural language questions to analyze the data. When provided with data modeling templates, the system can perform automatic joins based on key relationships and generate aggregations across datasets. It can also generate insights, visualizations, or SQL queries, while running LLMs and vectorDB locally instead of sending data to external APIs for data privacy and compliance.

## Example Queries

Users can ask questions like:

- What are the top 5 products by revenue?
- Show sales by region in a bar chart.
- Which month had the highest growth?
- Write a SQL query to calculate total revenue per customer.

The application responds with:

- Data insights and explanations.
- Generated visualizations.
- SQL queries or Python analysis code when requested.

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
