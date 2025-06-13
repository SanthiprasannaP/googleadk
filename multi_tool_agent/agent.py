import pandas as pd
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from typing import Dict, Union, List

# Load your CSV data
df = pd.read_csv("multi_tool_agent/claims.csv")

def run_query(query: str) -> Dict[str, Union[str, Dict]]:
    """Query the claims CSV data based on natural language input.
    
    Returns:
        Dict[str, Union[str, Dict]]: A dictionary containing:
            - status: 'success' or 'error'
            - data: The query results if successful
            - error_message: Error details if unsuccessful
    """
    try:
        if "average claim" in query.lower():
            avg_claims = df.groupby("product_type")["claim_amount"].mean()
            return {
                "status": "success",
                "data": avg_claims.to_dict(),
                "message": "Average claim amounts by product type"
            }
        elif "total claim" in query.lower():
            total_claims = df.groupby("product_type")["claim_amount"].sum()
            return {
                "status": "success",
                "data": total_claims.to_dict(),
                "message": "Total claim amounts by product type"
            }
        elif "count" in query.lower() or "number" in query.lower():
            claim_counts = df.groupby("product_type").size()
            return {
                "status": "success",
                "data": claim_counts.to_dict(),
                "message": "Number of claims by product type"
            }
        elif "summary" in query.lower() or "overview" in query.lower():
            summary = df.describe()
            return {
                "status": "success",
                "data": summary.to_dict(),
                "message": "Claims data summary"
            }
        else:
            return {
                "status": "error",
                "error_message": "I don't understand that query. I can help with average claims, total claims, count of claims, or summary statistics by product type."
            }
    except KeyError as e:
        return {
            "status": "error",
            "error_message": f"Column not found - {e}. Available columns: {list(df.columns)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error processing query: {str(e)}"
        }

def get_raw_data(limit: int = 10) -> Dict[str, Union[str, List[Dict]]]:
    """Returns a sample of the raw claims data.
    
    Returns:
        Dict[str, Union[str, List[Dict]]]: A dictionary containing:
            - status: 'success' or 'error'
            - data: List of dictionaries containing the sample data if successful
            - error_message: Error details if unsuccessful
    """
    try:
        sample_data = df.head(limit)
        return {
            "status": "success",
            "data": sample_data.to_dict('records'),
            "message": f"Sample of claims data (first {limit} rows)"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error retrieving data: {str(e)}"
        }

# Create FunctionTools
query_tool = FunctionTool(func=run_query)
raw_data_tool = FunctionTool(func=get_raw_data)

# Define the root agent for adk web
root_agent = Agent(
    name="claims_analysis_agent",
    model="gemini-2.0-flash",  # Specify the model to use
    tools=[query_tool, raw_data_tool],
    instruction="""You are a helpful agent that answers questions about insurance claims data from a CSV file.

When analyzing claims data:
- Use the 'run_query' tool to analyze the data and provide insights. This tool can handle:
  * Average claim amounts by product type
  * Total claim amounts by product type
  * Count of claims by product type
  * Summary statistics of the claims data

- Use the 'get_raw_data' tool to show sample data when requested.

Handle tool responses appropriately:
- If a tool returns a 'success' status, present the data to the user in a clear, formatted way
- If a tool returns an 'error' status, explain the error to the user and suggest alternative queries
- For complex analyses, you can use the tools sequentially to gather different insights

Always provide context and explanations with the data you present."""
)