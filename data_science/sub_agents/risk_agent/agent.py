# risk_agent/agent.py
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, load_artifacts
from google.adk.agents.callback_context import CallbackContext
import google.genai.types as types
import base64

from data_science.sub_agents.risk_agent.tools import (
    InsuranceRiskAgents,
    DashboardService,
    ReportService,
)

# Import the other agents
# from multi_tool_agent.explainability_agent.agent import root_agent as compliance_explainer
# from multi_tool_agent.dashboard_agent.agent import root_agent as dashboard_agent

risk_service = InsuranceRiskAgents()
dashboard_service = DashboardService()
report_service = ReportService()

# FunctionTool: Just predict and store
def predict_only(inputs: dict, tool_context=None) -> dict:
    input_data = inputs.get("input_data")
    if not input_data:
        return {"error": "Missing 'input_data' in inputs."}
    result = risk_service.predict_risk(input_data)
    if tool_context:
        tool_context.state["last_input"] = input_data
        tool_context.state["last_risk_output"] = result
    return {
        "risk_score": result.get("risk_score"),
        "confidence": result.get("confidence"),
        "note": "Do you want to explain the score?",
    }


def explain_risk(tool_context=None) -> dict:
    input_data = {}
    if tool_context and "last_input" in tool_context.state:
        input_data = tool_context.state["last_input"]
    result = risk_service.explain_risk(input_data)
    if tool_context:
        tool_context.state["last_explanation"] = result
    return result


def run_dashboard(tool_context):
    if not tool_context:
        return {"error": "Tool context missing."}
    result = dashboard_service.generate_summary(tool_context=tool_context)
    return {"dashboard": result}


def generate_report(summary: str, tool_context=None):
    if not tool_context or "last_risk_output" not in tool_context.state or "last_explanation" not in tool_context.state:
        return {"error": "Cannot generate report. Missing risk or explanation data from context."}
    
    risk_data = tool_context.state["last_risk_output"]
    explanation_data = tool_context.state["last_explanation"]
    
    pdf_artifact_dict = report_service.generate_pdf_report(risk_data, explanation_data, summary)
  

    # save the pdf locally
    with open(pdf_artifact_dict['file_name'], 'wb') as f:
        f.write(base64.b64decode(pdf_artifact_dict['file_data'].split(',')[-1]))
    
   
    return {"message": f"Successfully generated report: {pdf_artifact_dict['file_name']}", "file_name": pdf_artifact_dict['file_name']}


# Wrap the tools
predict_tool = FunctionTool(predict_only)
explain_tool = FunctionTool(explain_risk)
dashboard_tool = FunctionTool(run_dashboard)
report_tool = FunctionTool(generate_report)

# Root agent
root_agent = Agent(
    name="risk_modeling_agent",
    model="gemini-2.0-flash",
    tools=[predict_tool, explain_tool, dashboard_tool, report_tool],
    instruction="""
You are a calibrated insurance risk agent.

Step-by-step:
1.  Use `predict_only` to get a risk score. Expect input like: `{"input_data": {"age": 40, ...}}`.
2.  Ask if the user wants an explanation. If yes, call `explain_risk`. This tool requires no input arguments as it uses the last prediction's context.
3.  Ask if they want a visual dashboard. If yes, call `run_dashboard`.
4.  Finally, ask if they want a downloadable PDF report.
    a. If yes, first create a concise, natural-language summary of the findings (risk score, confidence, and key factors).
    b. Then, call the `generate_report` tool, passing the summary you created as the `summary` argument. For example: `generate_report(summary="The customer has a high risk score of 0.85, mainly due to their age and vehicle type.")`
    c. The tool will save the report as a downloadable artifact. Inform the user that the report has been generated and is ready for download.

Store conversation data in `tool_context.state`:
- "last_input"
- "last_risk_output"
- "last_explanation"

Guide the user through these steps clearly.
""",
    description="Agent that predicts insurance risk and provides explanations, dashboards, and PDF reports.",
)
