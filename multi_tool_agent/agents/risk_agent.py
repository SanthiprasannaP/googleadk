# agents/risk_agent.py
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from services.insurance_risk_service import InsuranceRiskAgents

risk_service = InsuranceRiskAgents()

predict_risk_tool = FunctionTool(risk_service.predict_risk)

risk_agent = Agent(
    name="risk_modeling_agent",
    model="gemini-2.0-flash",
    tools=[predict_risk_tool],
    instruction="Provide risk scores with confidence intervals",
    description="Production risk scoring engine"
)
