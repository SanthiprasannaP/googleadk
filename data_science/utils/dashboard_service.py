# services/dashboard_service.py
import json
import os
from datetime import datetime

class DashboardService:
    def generate_summary(self, tool_context):
        try:
            risk_data = tool_context.state["last_risk_output"]
            explain_data = tool_context.state["last_explanation"]
        except KeyError:
            return {"error": "Missing explanation or risk output in context."}

        summary = {
            "timestamp": datetime.now().isoformat(),
            "risk_score": risk_data.get("risk_score"),
            "confidence": risk_data.get("confidence"),
            "explanation": explain_data.get("nl_explanation"),
            "top_features": explain_data.get("factors"),
            "force_plot_path": explain_data.get("shap_force_plot_path"),
            "decision_plot_path": explain_data.get("decision_plot_path"),
            "model_version": explain_data.get("model_version"),
            "audit_id": explain_data.get("audit_id")
        }

        # Save summary JSON to dashboard_output/summary_output.json
        output_dir = os.path.join(os.getcwd(), "dashboard_output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "summary_output.json")

        try:
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=4)
        except Exception as e:
            return {"error": f"Failed to save dashboard summary: {str(e)}"}

        return summary
