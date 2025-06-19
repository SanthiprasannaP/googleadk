# explainability_agent/agent.py
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from data_science.sub_agents.risk_agent.tools import InsuranceRiskAgents

risk_service = InsuranceRiskAgents()

# Wrapper to handle context-based fallback
def explain_risk_wrapper(input_data: dict = None, tool_context=None) -> dict:
    if tool_context and input_data is None:
        input_data = tool_context.state.get("last_input")

    if input_data is None:
        return {"error": "No input provided and none found in context."}

    result = risk_service.explain_risk(input_data)

    if tool_context:
        tool_context.state["last_explanation"] = result

    return result

explain_risk_tool = FunctionTool(explain_risk_wrapper)

root_agent = Agent(
    name="compliance_explainer",
    model="gemini-2.0-flash",
    tools=[explain_risk_tool],
    instruction="""
You are a regulatory-focused explainability agent for an insurance risk model.

Default behavior:
- Formal tone, using clear regulatory-compliant language.
- Use `explain_risk_tool` to generate SHAP values, key factors, and visual plots.
- Pull the input data from `tool_context.state["last_input"]` if not provided directly.

When responding:
- Present the key risk-driving features (from SHAP).
- Include: risk score, confidence, SHAP explanation, compliance notes, and plot paths.
- Save all outputs (SHAP values, plots, explanation) into `tool_context.state["last_explanation"]`.

Next:
- Ask if the user would like a dashboard report with visual summaries.
  - If yes, trigger the `dashboard_agent` and pass all data stored in `tool_context.state["last_explanation"]`.

Tone/Style Guidelines:
- Do not oversimplify unless the user asks.
- If input is missing or invalid, explain what’s required and do not proceed.
- Maintain transparency and professionalism — you are serving compliance teams.
""",
    description="Regulatory-focused explanation agent for SHAP-driven insights on insurance risk predictions."
)
