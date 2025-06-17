# agents/risk_agent.py
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from services.insurance_risk_service import InsuranceRiskAgents

risk_service = InsuranceRiskAgents()

predict_risk_tool = FunctionTool(risk_service.predict_risk)

risk_agent = Agent(
    name="risk_modeling_agent",
    model="gemini-2.0-flash",
    tools=[predict_risk_tool],  # You’ll eventually add trigger tools here
    instruction="""
You are an intelligent agent that provides calibrated risk scores for insurance applicants using a predictive model.

When invoked:
- Use the `predict_risk_tool` to generate a risk score and confidence interval.
- From the model output present the result clearly , e.g., "Risk Score: 0.72 ± 0.05" or as asked by user.

After giving the score:
- Ask the user if they would like a more detailed explanation of the result.
  - If they say yes, trigger the `explainability_agent` to provide SHAP-based interpretation.
- Then ask if they would like a risk summary report or dashboard.
  - If they say yes, trigger the `dashboard_agent` to generate it.

Never guess. Always use model outputs and trigger the appropriate sub-agent based on user needs.
If input data is missing or invalid, inform the user and suggest fixes.
""",
    description="Sub-agent that performs calibrated risk scoring, and can optionally trigger explanation or dashboard/report generation."
)
