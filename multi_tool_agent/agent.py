import pandas as pd
from google.adk.agents import Agent
from google.adk.tools import tool

# Load your CSV data
df = pd.read_csv("claims.csv")

@tool
def run_query(query: str) -> str:
    """Query the claims CSV data based on natural language input."""
    if "average claim" in query.lower():
        try:
            avg_claims = df.groupby("product_type")["claim_amount"].mean()
            return f"Average claim amounts by product type:\n{avg_claims.to_string()}"
        except KeyError as e:
            return f"Error: Column not found - {e}. Available columns: {list(df.columns)}"
        except Exception as e:
            return f"Error processing query: {e}"
    
    elif "total claim" in query.lower():
        try:
            total_claims = df.groupby("product_type")["claim_amount"].sum()
            return f"Total claim amounts by product type:\n{total_claims.to_string()}"
        except KeyError as e:
            return f"Error: Column not found - {e}. Available columns: {list(df.columns)}"
        except Exception as e:
            return f"Error processing query: {e}"
    
    elif "count" in query.lower() or "number" in query.lower():
        try:
            claim_counts = df.groupby("product_type").size()
            return f"Number of claims by product type:\n{claim_counts.to_string()}"
        except Exception as e:
            return f"Error processing query: {e}"
    
    elif "summary" in query.lower() or "overview" in query.lower():
        try:
            summary = df.describe()
            return f"Claims data summary:\n{summary.to_string()}"
        except Exception as e:
            return f"Error processing query: {e}"
    
    return "Sorry, I don't understand that query. I can help with average claims, total claims, count of claims, or summary statistics by product type."

@tool
def get_raw_data(limit: int = 10) -> str:
    """Returns a sample of the raw claims data."""
    try:
        sample_data = df.head(limit)
        return f"Sample of claims data (first {limit} rows):\n{sample_data.to_string(index=False)}"
    except Exception as e:
        return f"Error retrieving data: {str(e)}"

# Define the root agent for adk web
root_agent = Agent(
    tools=[run_query, get_raw_data],
    instructions="You are a helpful agent that answers questions about insurance claims data from a CSV file. Use the run_query tool to analyze the data and provide insights, and the get_raw_data tool to show sample data when requested."
)