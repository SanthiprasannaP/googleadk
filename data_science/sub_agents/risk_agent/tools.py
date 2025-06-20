# services/insurance_risk_service.py
import joblib, shap, pandas as pd, numpy as np, uuid, os, hashlib # type: ignore
from datetime import datetime
import matplotlib.pyplot as plt
import os
import json 

class InsuranceRiskAgents:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'data', 'calibrated_risk_model.joblib')

        model_data = joblib.load(model_path)
        self.model = model_data['calibrated_model']
        self.background_data = model_data['background_data']

        self.base_pipeline = self.model.calibrated_classifiers_[0].estimator
        self.preprocessor = self.base_pipeline.named_steps['preprocessor']
        self.classifier = self.base_pipeline.named_steps['classifier']

        self.original_feature_names = (
            self.preprocessor.transformers_[0][2] +
            self.preprocessor.transformers_[1][2]
        )

        self.feature_names = self._get_feature_names()
        self.explainer = self._create_shap_explainer()
        self.model_version = "1.0.20240615"

    def _get_feature_names(self):
        numeric = self.preprocessor.transformers_[0][2]
        categorical = list(
            self.preprocessor.named_transformers_['cat']
            .get_feature_names_out(self.preprocessor.transformers_[1][2])
        )
        return numeric + categorical

    def _create_shap_explainer(self):
        return shap.TreeExplainer(
            self.classifier,
            data=self.background_data,
            feature_perturbation="interventional",
            model_output="probability"
        )

    def _validate_input(self, input_data: dict) -> bool:
        return all(feat in input_data for feat in self.original_feature_names)

    def _hash_input(self, input_data: dict) -> str:
        return hashlib.sha256(pd.Series(input_data).to_json().encode()).hexdigest()

    def predict_risk(self, input_data: dict) -> dict:
        if not self._validate_input(input_data):
            return {
                'error': f"Missing features. Required: {self.original_feature_names}",
                'model_version': self.model_version
            }
        try:
            df = pd.DataFrame([input_data])
            proba = self.model.predict_proba(df)[0, 1]
            return {
                'risk_score': float(proba),
                'confidence': float(abs(proba - 0.5) * 2),
                'model_version': self.model_version,
                'input_hash': self._hash_input(input_data)
            }
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'model_version': self.model_version
            }

    def explain_risk(self, input_data: dict, tool_context=None) -> dict:
        if not self._validate_input(input_data):
            return {
                'error': f"Missing features. Required: {self.original_feature_names}",
                'model_version': self.model_version
            }

        try:
            if tool_context:
                history = tool_context.state.get("conversation_history", [])
                history.append({
                    "input": input_data,
                    "timestamp": datetime.now().isoformat()
                })
                tool_context.state["conversation_history"] = history

            df = pd.DataFrame([input_data])
            processed = self.preprocessor.transform(df)
            shap_values = self.explainer.shap_values(processed)[0]
            risk_score = self.model.predict_proba(df)[0, 1]
            confidence = float(abs(risk_score - 0.5) * 2)

            top_indices = np.argsort(-np.abs(shap_values))[:4]
            top_factors = [
                f"{self.feature_names[i]} ({shap_values[i]:.2f}, {'increases' if shap_values[i] > 0 else 'reduces'} risk)"
                for i in top_indices
            ]

            risk_level = "High" if risk_score >= 0.5 else "Low"
            explanation = f"{risk_level} risk ({risk_score:.2f}) primarily due to: " + " + ".join(top_factors)

            force_plot_path = self._save_force_plot(shap_values)
            decision_plot_path = self._save_decision_plot(shap_values)
            audit_id = str(uuid.uuid4())

            if tool_context:
                audit_log = tool_context.state.get("audit_log", [])
                audit_log.append({
                    "audit_id": audit_id,
                    "input": input_data,
                    "result": top_factors,
                    "timestamp": datetime.now().isoformat(),
                    "input_hash": self._hash_input(input_data)
                })
                tool_context.state["audit_log"] = audit_log

            return {
                'risk_score': risk_score,
                'confidence': confidence,
                'factors': top_factors,
                'nl_explanation': explanation,
                'shap_force_plot_path': force_plot_path,
                'decision_plot_path': decision_plot_path,
                'compliance_note': "Explanation compliant with EU AI Act Article 13",
                'model_version': self.model_version,
                'input_hash': self._hash_input(input_data),
                'audit_id': audit_id
            }

        except Exception as e:
            return {
                'error': f"Explanation failed: {str(e)}",
                'model_version': self.model_version
            }

    
    def _save_force_plot(self, shap_values):
        shap.initjs()
        plot_html = shap.force_plot(
           self.explainer.expected_value,
           shap_values,
           feature_names=self.feature_names,
           matplotlib=False
        )
        plot_id = str(uuid.uuid4())

        shap_dir = os.path.join(os.getcwd(), "shap_outputs")
        os.makedirs(shap_dir, exist_ok=True)

        filename = f"shap_force_plot_{plot_id}.html"
        filepath = os.path.join(shap_dir, filename)
        shap.save_html(filepath, plot_html)
        return filepath

    

    def _save_decision_plot(self, shap_values):
        plot_id = str(uuid.uuid4())
        plt.figure(figsize=(12, 6))
        shap.decision_plot(
            self.explainer.expected_value,
            shap_values,
            self.feature_names,
            highlight=0,
            feature_order='importance',
            show=False,
            return_objects=False
        )
        plt.title("SHAP Decision Plot: Feature Contributions to Risk", fontsize=14)
        plt.xlabel("Model Output Probability", fontsize=12)
        plt.tight_layout()
        path = f"shap_decision_plot_{plot_id}.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path



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