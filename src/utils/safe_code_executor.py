import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import sys
from typing import Dict, Any, Tuple
import traceback

def execute_analysis_code(code: str, datasets: Dict[str, pd.DataFrame]) -> Tuple[str, Any, str]:
    """
    Executes Python code safely using a restricted local environment.
    
    Args:
        code: The Python code to execute.
        datasets: Dictionary of dataset names to pd.DataFrame objects.
        
    Returns:
        Tuple of (text_output, fig, error_message).
    """
    # Create the restricted execution environment
    local_env = {
        'pd': pd,
        'np': np,
        'plt': plt,
        'px': px,
        'go': go,
        'datasets': datasets,
        # Variables that the code can set to return useful objects
        'fig': None,        # For plotly/matplotlib figures
        'result_df': None,  # For returning a processed DataFrame
        'result_text': None # For returning simple text/number results
    }
    
    # Capture standard output
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    error_msg = None
    
    try:
        # Clear previous matplotlib figures
        plt.clf()
        plt.close('all')
        
        exec(code, {}, local_env)
    except Exception as e:
        error_msg = f"Error during execution:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        
    # Get printed output
    printed_output = redirected_output.getvalue()
    
    text_output = ""
    if printed_output:
        text_output += printed_output + "\n"
    
    if local_env.get('result_text') is not None:
        text_output += str(local_env['result_text'])
    
    # Extract figure if available
    fig = local_env.get('fig')
    
    # Extract dataframe if available, convert to text/markdown or just keep it
    result_df = local_env.get('result_df')
    if result_df is not None and isinstance(result_df, pd.DataFrame):
        # Limit to 20 rows for text output
        text_output += "\nData Result:\n" + result_df.head(20).to_markdown()
    elif result_df is not None and isinstance(result_df, pd.Series):
        text_output += "\nData Result:\n" + result_df.head(20).to_markdown()

    # If no explicit fig returned but matplotlib was used
    if fig is None and len(plt.get_fignums()) > 0:
        fig = plt.gcf()
        
    return text_output.strip(), fig, error_msg
