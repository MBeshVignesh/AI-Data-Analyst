# AI Data Analyst

A local AI data copilot that lets users explore datasets using natural language. Users can upload CSV,XLSX or JSON files and ask questions to get insights, generate plots, or even create SQL queries automatically.

## Example Queries

Users can ask questions like:

What are the top 5 products by revenue?
Show sales by region in a bar chart.
Which month had the highest growth?
Write a SQL query to calculate total revenue per customer.

The application responds with:

Data insights and explanations.
Generated visualizations.
SQL queries or Python analysis code when requested.

## Technology Stack

Programming & Data Processing
Python
Pandas
NumPy

Visualization
Plotly
Matplotlib

AI & LLM Integration
Ollama - llama3.2
LangChain

Vector Database
ChromaDB

User Interface
Gradio

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
