# AI Data Analyst

A local AI Data Analyst application that allows users to analyze datasets using natural language. Built with Python, Gradio, Pandas, Plotly, LangChain, ChromaDB, and Ollama.

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
