# risk_agent/agent.py
"""Risk Assessment Agent for Insurance Analytics.

This module defines the comprehensive risk assessment agent that provides
insurance risk scoring, explainability, and impact simulation capabilities.
It integrates multiple services to deliver end-to-end risk analysis for
insurance applications.

The agent provides the following core services:
1. Risk Prediction: Calculate risk scores for insurance applications
2. Risk Explanation: Provide SHAP-based explanations for risk scores
3. Dashboard Generation: Create visual summaries of risk analysis
4. Report Generation: Generate comprehensive PDF reports
5. Impact Simulation: Simulate the effect of changes on risk scores
6. What-if Analysis: Compare multiple scenarios and their outcomes

Key Components:
- InsuranceRiskAgents: Core risk assessment service
- DashboardService: Visual dashboard generation
- ReportService: PDF report creation
- ImpactSimulatorService: Scenario simulation and analysis

Dependencies:
- Google ADK for agent framework
- SHAP for explainability
- FPDF for report generation
- Various risk modeling tools
"""

from typing import Dict, Any, Optional
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, load_artifacts
from google.adk.agents.callback_context import CallbackContext
import google.genai.types as types
import base64

from data_science.sub_agents.risk_agent.tools import (
    InsuranceRiskAgents,
    DashboardService,
    ReportService,
    ImpactSimulatorService,
)

# Import the other agents
# from multi_tool_agent.explainability_agent.agent import root_agent as compliance_explainer
# from multi_tool_agent.dashboard_agent.agent import root_agent as dashboard_agent

# Initialize service instances for risk assessment
risk_service = InsuranceRiskAgents()
dashboard_service = DashboardService()
report_service = ReportService()
simulator_service = ImpactSimulatorService()


def predict_only(inputs: Dict[str, Any], tool_context=None) -> Dict[str, Any]:
    """Calculate risk score for insurance application data.
    
    This function takes customer and policy data and calculates a risk score
    using the trained machine learning model. It provides both the risk score
    and confidence level for the prediction.
    
    Args:
        inputs: Dictionary containing 'input_data' with customer/policy information.
                Expected fields include age, income, vehicle_age, vehicle_value,
                premium_amount, vehicle_brand, occupation, city, policy_type, gender.
        tool_context: Optional context for storing state between function calls.
    
    Returns:
        Dict containing:
            - risk_score: Float between 0 and 1 indicating risk level
            - confidence: Float indicating prediction confidence
            - note: Suggestion for next steps (explanation)
            - error: Error message if prediction fails
    
    Example:
        >>> result = predict_only({
        ...     "input_data": {
        ...         "age": 35,
        ...         "income": 75000,
        ...         "vehicle_age": 2,
        ...         "vehicle_value": 50000,
        ...         "premium_amount": 1200,
        ...         "vehicle_brand": "Toyota",
        ...         "occupation": "engineer",
        ...         "city": "Metropolis",
        ...         "policy_type": "Comprehensive",
        ...         "gender": "M"
        ...     }
        ... })
    """
    input_data = inputs.get("input_data")
    if not input_data:
        return {"error": "Missing 'input_data' in inputs."}
    
    # Calculate risk score using the trained model
    result = risk_service.predict_risk(input_data)
    
    # Store results in context for potential use by other functions
    if tool_context:
        tool_context.state["last_input"] = input_data
        tool_context.state["last_risk_output"] = result
    
    return {
        "risk_score": result.get("risk_score"),
        "confidence": result.get("confidence"),
        "note": "Do you want to explain the score?",
    }


def explain_risk(tool_context=None) -> Dict[str, Any]:
    """Generate SHAP-based explanation for the most recent risk score.
    
    This function provides detailed explanations of risk scores using SHAP
    (SHapley Additive exPlanations) values. It identifies the key factors
    contributing to the risk assessment and generates visualizations.
    
    Args:
        tool_context: Context containing the last input data and risk output.
    
    Returns:
        Dict containing:
            - risk_score: The calculated risk score
            - confidence: Prediction confidence level
            - factors: Top contributing factors to the risk score
            - nl_explanation: Natural language explanation of the risk
            - shap_force_plot_path: Path to SHAP force plot visualization
            - decision_plot_path: Path to SHAP decision plot visualization
            - compliance_note: EU AI Act compliance information
            - model_version: Version of the risk model used
            - input_hash: Hash of input data for audit purposes
            - audit_id: Unique identifier for audit trail
            - error: Error message if explanation fails
    
    Note:
        This function requires that predict_only has been called first
        to populate the tool_context with input data.
    """
    input_data = {}
    if tool_context and "last_input" in tool_context.state:
        input_data = tool_context.state["last_input"]
    
    # Generate comprehensive risk explanation with SHAP analysis
    result = risk_service.explain_risk(input_data)
    
    # Store explanation results in context
    if tool_context:
        tool_context.state["last_explanation"] = result
    
    return result


def run_dashboard(tool_context) -> Dict[str, Any]:
    """Generate visual dashboard summary of risk analysis.
    
    This function creates a comprehensive dashboard that summarizes
    the risk assessment results, including visualizations and key metrics.
    
    Args:
        tool_context: Context containing risk analysis results and state.
    
    Returns:
        Dict containing:
            - dashboard: Dashboard summary and visualizations
            - error: Error message if dashboard generation fails
    
    Note:
        This function requires previous risk analysis results to be
        available in the tool_context.
    """
    if not tool_context:
        return {"error": "Tool context missing."}
    
    # Generate dashboard summary using the dashboard service
    result = dashboard_service.generate_summary(tool_context=tool_context)
    return {"dashboard": result}


def generate_report(summary: str, tool_context=None) -> Dict[str, Any]:
    """Generate comprehensive PDF report of risk analysis.
    
    This function creates a detailed PDF report containing the risk assessment
    results, explanations, visualizations, and recommendations. The report
    is saved locally and can be used for compliance and audit purposes.
    
    Args:
        summary: Additional summary text to include in the report.
        tool_context: Context containing risk and explanation data.
    
    Returns:
        Dict containing:
            - message: Success message with file information
            - file_name: Name of the generated PDF file
            - error: Error message if report generation fails
    
    Note:
        This function requires both risk assessment and explanation results
        to be available in the tool_context from previous function calls.
    """
    if not tool_context or "last_risk_output" not in tool_context.state or "last_explanation" not in tool_context.state:
        return {"error": "Cannot generate report. Missing risk or explanation data from context."}
    
    # Get risk and explanation data from context
    risk_data = tool_context.state["last_risk_output"]
    explanation_data = tool_context.state["last_explanation"]
    
    # Generate PDF report using the report service
    pdf_artifact_dict = report_service.generate_pdf_report(risk_data, explanation_data, summary)
  
    # Save the PDF locally for access
    with open(pdf_artifact_dict['file_name'], 'wb') as f:
        f.write(base64.b64decode(pdf_artifact_dict['file_data'].split(',')[-1]))
    
    return {
        "message": f"Successfully generated report: {pdf_artifact_dict['file_name']}", 
        "file_name": pdf_artifact_dict['file_name']
    }


def simulate_impact(inputs: Dict[str, Any], tool_context=None) -> Dict[str, Any]:
    """Simulate impact of changes on risk score and metrics.
    
    This function allows users to simulate how changes in customer data,
    policy parameters, or risk mitigation strategies would affect the
    overall risk score and related metrics.
    
    Args:
        inputs: Dictionary containing:
            - base_data: Original customer/policy data
            - scenario_changes: Dictionary describing changes to simulate
        tool_context: Optional context for storing simulation results.
    
    Returns:
        Dict containing:
            - original_risk_score: Risk score before changes
            - new_risk_score: Risk score after changes
            - risk_change: Absolute change in risk score
            - risk_change_percentage: Percentage change in risk score
            - scenario_summary: Summary of changes made
            - recommendations: Suggested actions based on simulation
            - error: Error message if simulation fails
    
    Example:
        >>> result = simulate_impact({
        ...     "base_data": {"age": 35, "vehicle_age": 5, ...},
        ...     "scenario_changes": {
        ...         "risk_mitigation": ["add_telematics", "defensive_driving"]
        ...     }
        ... })
    """
    base_data = inputs.get("base_data")
    scenario_changes = inputs.get("scenario_changes", {})
    
    if not base_data:
        return {"error": "Missing 'base_data' in inputs."}
    
    # Get original risk score if not provided
    if 'risk_score' not in base_data:
        original_result = risk_service.predict_risk(base_data)
        if 'error' not in original_result:
            base_data['risk_score'] = original_result.get('risk_score', 0.5)
    
    # Run impact simulation
    result = simulator_service.simulate_impact(base_data, scenario_changes)
    
    # Store simulation results in context
    if tool_context:
        tool_context.state["simulation_result"] = result
    
    return result


def run_what_if_analysis(inputs: Dict[str, Any], tool_context=None) -> Dict[str, Any]:
    """Run multiple what-if scenarios and compare results.
    
    This function allows users to compare multiple scenarios simultaneously
    to understand the relative impact of different changes on risk scores.
    It's useful for strategic decision-making and policy optimization.
    
    Args:
        inputs: Dictionary containing:
            - base_data: Original customer/policy data
            - scenarios: List of scenario dictionaries to compare
        tool_context: Optional context for storing analysis results.
    
    Returns:
        Dict containing:
            - base_scenario: Original risk assessment
            - scenario_comparisons: List of scenario results
            - best_scenario: Scenario with lowest risk score
            - worst_scenario: Scenario with highest risk score
            - recommendations: Strategic recommendations
            - error: Error message if analysis fails
    
    Example:
        >>> result = run_what_if_analysis({
        ...     "base_data": {"age": 35, "vehicle_age": 5, ...},
        ...     "scenarios": [
        ...         {"name": "Telematics Only", "changes": {...}},
        ...         {"name": "Defensive Driving", "changes": {...}},
        ...         {"name": "Combined Approach", "changes": {...}}
        ...     ]
        ... })
    """
    base_data = inputs.get("base_data")
    scenarios = inputs.get("scenarios", [])
    
    if not base_data or not scenarios:
        return {"error": "Missing 'base_data' or 'scenarios' in inputs."}
    
    # Get original risk score if not provided
    if 'risk_score' not in base_data:
        original_result = risk_service.predict_risk(base_data)
        if 'error' not in original_result:
            base_data['risk_score'] = original_result.get('risk_score', 0.5)
    
    # Run comprehensive what-if analysis
    result = simulator_service.run_what_if_analysis(base_data, scenarios)
    
    # Store analysis results in context
    if tool_context:
        tool_context.state["what_if_analysis"] = result
    
    return result


def get_simulation_scenarios(inputs: Optional[Dict[str, Any]] = None, tool_context=None) -> Dict[str, Any]:
    """Get available simulation scenarios and their descriptions.
    
    This function provides information about all available simulation scenarios,
    including their descriptions, options, and usage examples. It helps users
    understand what types of changes they can simulate.
    
    Args:
        inputs: Optional dictionary (not used in current implementation).
        tool_context: Optional context (not used in current implementation).
    
    Returns:
        Dict containing:
            - available_scenarios: Dictionary of scenario categories and options
            - usage_examples: Example inputs for different simulation types
    
    Example:
        >>> scenarios = get_simulation_scenarios()
        >>> print(scenarios["available_scenarios"]["risk_mitigation"])
    """
    scenarios = {
        "premium_adjustment": {
            "description": "Adjust premium amounts and coverage levels",
            "options": ["increase_premium", "decrease_premium", "change_coverage"]
        },
        "risk_mitigation": {
            "description": "Implement risk reduction strategies",
            "options": ["add_telematics", "defensive_driving", "vehicle_upgrade"]
        },
        "policy_changes": {
            "description": "Modify policy terms and conditions",
            "options": ["change_deductible", "add_riders", "modify_coverage_limits"]
        },
        "behavioral_changes": {
            "description": "Simulate changes in customer behavior",
            "options": ["reduce_mileage", "change_vehicle", "improve_credit_score"]
        }
    }
    
    return {
        "available_scenarios": scenarios,
        "usage_examples": {
            "single_simulation": {
                "base_data": {
                    "age": 45,
                    "income": 85000,
                    "vehicle_age": 3,
                    "vehicle_value": 75000,
                    "premium_amount": 12000,
                    "vehicle_brand": "Toyota",
                    "occupation": "engineer",
                    "city": "Metropolis",
                    "policy_type": "Comprehensive",
                    "gender": "M"
                },
                "scenario_changes": {"risk_mitigation": ["add_telematics", "defensive_driving"]}
            },
            "what_if_analysis": {
                "base_data": {
                    "age": 45,
                    "income": 85000,
                    "vehicle_age": 3,
                    "vehicle_value": 75000,
                    "premium_amount": 12000,
                    "vehicle_brand": "Toyota",
                    "occupation": "engineer",
                    "city": "Metropolis",
                    "policy_type": "Comprehensive",
                    "gender": "M"
                },
                "scenarios": [
                    {"name": "Telematics Only", "changes": {"risk_mitigation": ["add_telematics"]}},
                    {"name": "Defensive Driving Only", "changes": {"risk_mitigation": ["defensive_driving"]}},
                    {"name": "Vehicle Upgrade", "changes": {"risk_mitigation": ["vehicle_upgrade"]}},
                    {"name": "Combined Approach", "changes": {"risk_mitigation": ["add_telematics", "defensive_driving"]}}
                ]
            }
        }
    }


# Create FunctionTool wrappers for all risk assessment functions
predict_tool = FunctionTool(predict_only)
explain_tool = FunctionTool(explain_risk)
dashboard_tool = FunctionTool(run_dashboard)
report_tool = FunctionTool(generate_report)
simulate_tool = FunctionTool(simulate_impact)
what_if_tool = FunctionTool(run_what_if_analysis)
scenarios_tool = FunctionTool(get_simulation_scenarios)

# Main risk assessment agent with comprehensive capabilities
root_agent = Agent(
    name="risk_modeling_agent",
    model="gemini-2.0-flash",
    tools=[predict_tool, explain_tool, dashboard_tool, report_tool, simulate_tool, what_if_tool, scenarios_tool],
    instruction="""
You are a comprehensive insurance risk agent with impact simulation capabilities.

**Available Services:**

1. **Risk Assessment** (`predict_only`): Calculate risk scores for insurance applications
2. **Risk Explanation** (`explain_risk`): Provide SHAP-based explanations for risk scores
3. **Dashboard** (`run_dashboard`): Generate visual dashboard summaries
4. **Report Generation** (`generate_report`): Create comprehensive PDF reports
5. **Impact Simulation** (`simulate_impact`): Simulate changes and their effects,Compare multiple scenarios takes only 1 input. if user gives a list of scenarios, please call this multiple times.
6. **What-if Analysis** (`run_what_if_analysis`): Compare multiple scenarios takes only 1 input. if user gives a list of scenarios, please call this multiple times.
7. **Scenario Information** (`get_simulation_scenarios`): Get available simulation options,Compare multiple scenarios takes only 1 input. if user gives a list of scenarios, please call this multiple times.

**Workflow:**
1. Start with risk assessment using customer/policy data
2. Get explanations to understand risk factors
3. Generate dashboards or reports as needed
4. Use simulations to explore optimization opportunities


**Input Format:**
Provide customer and policy data including age, income, vehicle information,
premium amounts, and other relevant factors for comprehensive risk analysis.
""",
)
