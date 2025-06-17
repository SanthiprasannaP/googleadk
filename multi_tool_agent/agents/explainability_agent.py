# agents/explainability_agent.py
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from services.insurance_risk_service import InsuranceRiskAgents

risk_service = InsuranceRiskAgents()
explain_risk_tool = FunctionTool(risk_service.explain_risk)

explain_agent = Agent(
    name="compliance_explainer",
    model="gemini-2.0-pro",
    tools=[explain_risk_tool],
    instruction="Generate regulator-friendly explanations with SHAP plots",
    description="Enterprise explainability service"
)
