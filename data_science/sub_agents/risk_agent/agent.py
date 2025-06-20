# risk_agent/agent.py
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from google.adk.tools import load_artifacts

from data_science.sub_agents.risk_agent.tools import (
    InsuranceRiskAgents,
    DashboardService
)

# Import the other agents
# from multi_tool_agent.explainability_agent.agent import root_agent as compliance_explainer
# from multi_tool_agent.dashboard_agent.agent import root_agent as dashboard_agent

risk_service = InsuranceRiskAgents()
dashboardService = DashboardService()

# FunctionTool: Just predict and store
def predict_only(inputs: dict, tool_context=None) -> dict:
    input_data = inputs.get("input_data")
    if not input_data:
        return {"error": "Missing 'input_data' in inputs."}

    result = risk_service.predict_risk(input_data)

    # Save prediction to context
    if tool_context:
        tool_context.state["last_input"] = input_data
        tool_context.state["last_risk_output"] = result

    return {
        "risk_score": result.get("risk_score"),
        "confidence": result.get("confidence"),
        "note": "Do you want to explain the score?"
    }


def explain_risk(tool_context=None) -> dict:
    if tool_context and "last_input" in tool_context.state:
        # Use the last input from the tool context if available
        input_data = tool_context.state["last_input"]
    elif "input_data" in input_data:
      input_data = input_data.get("input_data")

    result = risk_service.explain_risk(input_data)

    if tool_context:
        tool_context.state["last_explanation"] = result

    return result



def run_dashboard(tool_context):
    if not tool_context:
        return {"error": "Tool context missing."}
    result = dashboardService.generate_summary(tool_context=tool_context)
    return {
        "dashboard": result
    }

# Wrap the tools
predict_tool = FunctionTool(predict_only)
explain_tool = FunctionTool(explain_risk)
dashboard_tool = FunctionTool(run_dashboard)

# Root agent
root_agent = Agent(
    name="risk_modeling_agent",
    model="gemini-2.0-flash",
    tools=[predict_tool, explain_tool, load_artifacts, dashboard_tool],
    instruction="""
You are a calibrated insurance risk agent.

When receiving user input to predict risk, expect the input data under the key 'input_data' as a dictionary of features, for example:

  "input_data": {
    "age": 40,
    "income": 85900,
    "vehicle_age": 3,
    "vehicle_value": 75000,
    "premium_amount": 1200,
    "vehicle_brand": "Toyota",
    "occupation": "engineer",
    "city": "Metropolis",
    "policy_type": "Comprehensive",
    "gender": "M"
  }


Step-by-step:
1. Use `predict_insurance_risk` tool with the 'input_data' dictionary to return a risk score.
2. Ask if the user wants an explanation — if yes, use `explain_risk`.
2a. Use `explain_risk` to generate SHAP values, key factors, and visual plots. input data in the same format as above with tool context.
3. Then ask if they want a visual dashboard — if yes, use `run_dashboard`.

Store:
- Input → tool_context.state["last_input"]
- Prediction → tool_context.state["last_risk_output"]
- Explanation → tool_context.state["last_explanation"
- Dashboard → tool_context.state["dashboard"]

Be clear, concise, and guide the user through these steps.
""",
    description="Agent that predicts insurance risk and optionally explains and visualizes it."
)
