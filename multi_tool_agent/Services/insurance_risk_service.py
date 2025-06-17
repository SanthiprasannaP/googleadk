# services/insurance_risk_service.py
import joblib, shap, pandas as pd, numpy as np, uuid, os
from datetime import datetime
import matplotlib.pyplot as plt
import os

class InsuranceRiskAgents:
    def __init__(self):
        # Load model and background data
        model_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'calibrated_risk_model.joblib')
        model_path = os.path.abspath(model_path)  
        model_data = joblib.load(model_path)
        self.model = model_data['calibrated_model']
        self.background_data = model_data['background_data']
        
        # Access base pipeline
        self.base_pipeline = self.model.calibrated_classifiers_[0].estimator
        self.preprocessor = self.base_pipeline.named_steps['preprocessor']
        self.classifier = self.base_pipeline.named_steps['classifier']
        
        # Original input features
        self.original_feature_names = (
            self.preprocessor.transformers_[0][2] +  # numeric
            self.preprocessor.transformers_[1][2]    # categorical
        )
        
        # Transformed feature names
        self.feature_names = self._get_feature_names()
        
        # SHAP explainer
        self.explainer = self._create_shap_explainer()
        
        # Version
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
                'model_version': self.model_version
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
            # Log conversation history
            if tool_context:
                history = tool_context.state.get("conversation_history", [])
                history.append({
                    "input": input_data,
                    "timestamp": datetime.now().isoformat()
                })
                tool_context.state["conversation_history"] = history

            # Prepare input
            df = pd.DataFrame([input_data])
            processed = self.preprocessor.transform(df)
            shap_values = self.explainer.shap_values(processed)[0]
            risk_score = self.model.predict_proba(df)[0, 1]
            confidence = float(abs(risk_score - 0.5) * 2)

            # Top features
            top_indices = np.argsort(-np.abs(shap_values))[:4]
            top_factors = [
                f"{self.feature_names[i]} ({shap_values[i]:.2f}, {'increases' if shap_values[i] > 0 else 'reduces'} risk)"
                for i in top_indices
            ]

            # Natural language explanation
            risk_level = "High" if risk_score >= 0.5 else "Low"
            explanation = f"{risk_level} risk ({risk_score:.2f}) primarily due to: " + " + ".join(top_factors)

            # Save force plot (HTML)
            force_plot_path = self._save_force_plot(shap_values)

            # Save enhanced decision plot (PNG)
            decision_plot_path = self._save_decision_plot(shap_values)

            # Generate audit ID
            audit_id = str(uuid.uuid4())
            if tool_context:
                audit_log = tool_context.state.get("audit_log", [])
                audit_log.append({
                    "audit_id": audit_id,
                    "input": input_data,
                    "result": top_factors,
                    "timestamp": datetime.now().isoformat()
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
                'audit_id': audit_id
            }

        except Exception as e:
            return {
                'error': f"Explanation failed: {str(e)}",
                'model_version': self.model_version
            }

    def _save_force_plot(self, shap_values):
        """Save SHAP force plot as HTML file"""
        shap.initjs()
        plot_html = shap.force_plot(
            self.explainer.expected_value,
            shap_values,
            feature_names=self.feature_names,
            matplotlib=False
        )
        plot_id = str(uuid.uuid4())
        filename = f"shap_force_plot_{plot_id}.html"
        filepath = os.path.join(".", filename)
        shap.save_html(filepath, plot_html)
        return filepath

    def _save_decision_plot(self, shap_values):
        """Save enhanced SHAP decision plot as PNG"""
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
