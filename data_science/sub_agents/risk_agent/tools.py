# services/insurance_risk_service.py
import joblib, shap, pandas as pd, numpy as np, uuid, os, hashlib # type: ignore
from datetime import datetime
import matplotlib.pyplot as plt
import os
import json 
from fpdf import FPDF # type: ignore
from google.adk.agents.callback_context import CallbackContext
import base64
import io

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
        plot_id = str(uuid.uuid4())
        plt.figure(figsize=(12, 6))
        shap.force_plot(
            self.explainer.expected_value,
            shap_values,
            self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title("SHAP Force Plot: Feature Contributions to Risk", fontsize=14)
        plt.xlabel("SHAP Value", fontsize=12)
        plt.tight_layout()
        shap_dir = os.path.join(os.getcwd(), "shap_outputs")
        os.makedirs(shap_dir, exist_ok=True)
        file_name = f"shap_force_plot_{plot_id}.png"
        path = os.path.join(shap_dir, file_name)
        plt.savefig(path, bbox_inches='tight')
        plt.close()
     
        return file_name
       
    

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

class ReportService:
    def generate_pdf_report(self, risk_data: dict, explanation_data: dict, summary_text: str) -> str:
        ## json load summary output
        json_path = os.path.join(os.getcwd(), "dashboard_output", "summary_output.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                summary_data = json.load(f)
        else:
            summary_data = {
                "risk_score": risk_data.get("risk_score", 0),
                "confidence": risk_data.get("confidence", 0),
                "top_features": explanation_data.get("factors", []),
                "explanation": explanation_data.get("nl_explanation", ""),
                "force_plot_path": explanation_data.get("shap_force_plot_path", ""),
                "decision_plot_path": explanation_data.get("decision_plot_path", ""),
                "model_version": explanation_data.get("model_version", ""),
                "audit_id": explanation_data.get("audit_id", "")
            }

        # Create PDF report
        pdf = FPDF()
        pdf.add_page()

        # Header section
        pdf.set_font("Arial", 'B', 18)
        pdf.set_text_color(0, 70, 130)  # Blue branding
        pdf.cell(0, 15, 'SecureDrive Insurance', 0, 1, 'C')
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 8, 'AI-Powered Risk Intelligence', 0, 1, 'C')
        pdf.ln(5)

        # Title
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, 'Insurance Risk Assessment Report', 0, 1, 'C')
        pdf.ln(10)

        # Summary
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 10, summary_text)
        pdf.ln(10)

        # Key Metrics
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 10, 'Key Metrics', 0, 1)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, f"Risk Score: {risk_data.get('risk_score', 'N/A'):.2f}", 0, 1)
        pdf.cell(0, 10, f"Confidence: {risk_data.get('confidence', 'N/A'):.2f}", 0, 1)
        pdf.ln(5)

        # Risk Factors
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 10, 'Top Risk Factors', 0, 1)
        pdf.set_font("Arial", '', 10)
        for factor in explanation_data.get('factors', []):
            pdf.cell(0, 10, f"- {factor}", 0, 1)
        pdf.ln(10)

        # Decision Plot
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 10, 'SHAP Decision Plot', 0, 1)
        pdf.ln(5)

        image_path = summary_data.get('decision_plot_path', '')
        if os.path.exists(image_path):
            pdf.image(image_path, x=10, w=190)
        else:
            pdf.cell(0, 10, "Decision plot image not found.", 0, 1)
        pdf.ln(10)

        # Force Plot
        image_path = "shap_outputs/" + summary_data.get('force_plot_path', '')
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 10, 'SHAP Force Plot', 0, 1)
        pdf.ln(5)
        if os.path.exists(image_path):
            pdf.image(image_path, x=10, w=190)
        else:
            pdf.cell(0, 10, "Force plot image not found.", 0, 1)
        pdf.ln(10)

        # Footer
        pdf.set_y(-30)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 6, 'Generated by SecureDrive Insurance AI Risk Assessment System', 0, 1, 'C')

        # Return as base64 data URI
        pdf_output = pdf.output(dest='S')
        pdf_base64 = base64.b64encode(pdf_output).decode('utf-8')

        report_artifact = {
            "file_name": "risk_report.pdf",
            "file_data": f"data:application/pdf;base64,{pdf_base64}",
            "file_type": "application/pdf",
        }
        return report_artifact



class ImpactSimulatorService:
    def __init__(self):
        self.simulation_scenarios = {
            'premium_adjustment': ['increase_premium', 'decrease_premium', 'change_coverage'],
            'risk_mitigation': ['add_telematics', 'defensive_driving', 'vehicle_upgrade'],
            'policy_changes': ['change_deductible', 'add_riders', 'modify_coverage_limits'],
            'behavioral_changes': ['reduce_mileage', 'change_vehicle', 'improve_credit_score']
        }
        # Initialize the risk service to use the actual model
        self.risk_service = InsuranceRiskAgents()
    
    def simulate_impact(self, base_data: dict, scenario_changes: dict) -> dict:
        """Simulate impact of changes on risk score and metrics"""
        
        # Create modified data based on scenario
        modified_data = self._apply_scenario_changes(base_data, scenario_changes)
        
        # Calculate new risk score using the actual risk model
        new_risk_score = self._calculate_new_risk_score(modified_data)
        
        # Calculate impact metrics
        impact_metrics = self._calculate_impact_metrics(base_data, modified_data, new_risk_score)
        
        # Generate recommendations
        recommendations = self._generate_simulation_recommendations(scenario_changes, impact_metrics)
        
        return {
            'original_risk_score': self._get_original_risk_score(base_data),
            'new_risk_score': new_risk_score,
            'risk_score_change': new_risk_score - self._get_original_risk_score(base_data),
            'impact_metrics': impact_metrics,
            'scenario_changes': scenario_changes,
            'recommendations': recommendations,
            'confidence_level': self._calculate_confidence_level(scenario_changes),
            'simulation_id': self._generate_simulation_id()
        }
    
    def _apply_scenario_changes(self, base_data: dict, scenario_changes: dict) -> dict:
        """Apply scenario changes to base data using original model structure"""
        modified_data = base_data.copy()
        
        for change_type, changes in scenario_changes.items():
            if change_type == 'premium_adjustment':
                if 'increase_premium' in changes:
                    modified_data['premium_amount'] = modified_data.get('premium_amount', 0) * 1.2
                elif 'decrease_premium' in changes:
                    modified_data['premium_amount'] = modified_data.get('premium_amount', 0) * 0.8
            
            elif change_type == 'risk_mitigation':
                if 'add_telematics' in changes:
                    # Add telematics as a new field (will be handled in risk calculation)
                    modified_data['has_telematics'] = True
                elif 'defensive_driving' in changes:
                    # Add defensive driving as a new field
                    modified_data['has_defensive_driving'] = True
                elif 'vehicle_upgrade' in changes:
                    # Upgrade vehicle (reduce age, increase value)
                    modified_data['vehicle_age'] = max(0, modified_data.get('vehicle_age', 5) - 2)
                    modified_data['vehicle_value'] = modified_data.get('vehicle_value', 50000) * 1.3
                    # Upgrade to a safer brand
                    modified_data['vehicle_brand'] = 'Toyota'  # Generally lower risk
            
            elif change_type == 'policy_changes':
                if 'change_deductible' in changes:
                    # This would affect premium calculation, not risk score directly
                    modified_data['deductible'] = modified_data.get('deductible', 1000) + changes.get('deductible_change', 500)
                elif 'add_riders' in changes:
                    # Add additional coverage options
                    modified_data['has_roadside_assistance'] = True
                    modified_data['has_rental_car_coverage'] = True
            
            elif change_type == 'behavioral_changes':
                if 'reduce_mileage' in changes:
                    # Reduce annual mileage (would need to be added to model)
                    modified_data['annual_mileage'] = modified_data.get('annual_mileage', 12000) * 0.7
                elif 'change_vehicle' in changes:
                    # Change to a safer vehicle brand
                    modified_data['vehicle_brand'] = 'Toyota'  # Generally lower risk
                    modified_data['vehicle_value'] = modified_data.get('vehicle_value', 50000) * 1.1
                elif 'improve_credit_score' in changes:
                    # Improve credit score (would need to be added to model)
                    modified_data['credit_score'] = min(850, modified_data.get('credit_score', 650) + 50)
        
        return modified_data
    
    def _calculate_new_risk_score(self, modified_data: dict) -> float:
        """Calculate new risk score using the actual risk model"""
        try:
            # Use the actual risk model to calculate new score
            # Remove any fields that aren't in the original model structure
            model_input = self._prepare_model_input(modified_data)
            
            # Get prediction from the actual risk model
            result = self.risk_service.predict_risk(model_input)
            
            if 'error' in result:
                # Fallback to simplified calculation if model fails
                return self._fallback_risk_calculation(modified_data)
            
            return result.get('risk_score', 0.5)
            
        except Exception as e:
            # Fallback to simplified calculation
            return self._fallback_risk_calculation(modified_data)
    
    def _prepare_model_input(self, data: dict) -> dict:
        """Prepare input data for the original risk model"""
        # Original model expects these fields:
        original_fields = [
            'age', 'income', 'vehicle_age', 'vehicle_value', 'premium_amount',
            'vehicle_brand', 'occupation', 'city', 'policy_type', 'gender'
        ]
        
        # Filter to only include original model fields
        model_input = {}
        for field in original_fields:
            if field in data:
                model_input[field] = data[field]
        
        return model_input
    
    def _fallback_risk_calculation(self, modified_data: dict) -> float:
        """Fallback risk calculation when model fails"""
        base_score = modified_data.get('risk_score', 0.5)
        adjustments = 0.0
        
        # Apply adjustments based on changes
        if modified_data.get('has_telematics', False):
            adjustments -= 0.1
        
        if modified_data.get('has_defensive_driving', False):
            adjustments -= 0.15
        
        if modified_data.get('vehicle_age', 5) < 3:
            adjustments -= 0.05
        
        if modified_data.get('vehicle_brand') == 'Toyota':
            adjustments -= 0.08
        
        if modified_data.get('annual_mileage', 12000) < 8000:
            adjustments -= 0.12
        
        new_score = max(0.01, min(0.99, base_score + adjustments))
        return round(new_score, 4)
    
    def _calculate_impact_metrics(self, base_data: dict, modified_data: dict, new_risk_score: float) -> dict:
        """Calculate comprehensive impact metrics"""
        original_score = self._get_original_risk_score(base_data)
        score_change = new_risk_score - original_score
        
        # Premium impact
        original_premium = base_data.get('premium_amount', 1000)
        new_premium = self._calculate_new_premium(original_premium, score_change)
        premium_change = new_premium - original_premium
        
        # Risk category change
        original_category = self._get_risk_category(original_score)
        new_category = self._get_risk_category(new_risk_score)
        
        # Confidence level
        confidence_change = abs(new_risk_score - 0.5) * 2 - abs(original_score - 0.5) * 2
        
        return {
            'premium_change': round(premium_change, 2),
            'premium_change_percentage': round((premium_change / original_premium) * 100, 2),
            'risk_category_change': f"{original_category} → {new_category}",
            'confidence_change': round(confidence_change, 4),
            'score_improvement': score_change < 0,
            'impact_magnitude': self._get_impact_magnitude(abs(score_change))
        }
    
    def _calculate_new_premium(self, original_premium: float, score_change: float) -> float:
        """Calculate new premium based on risk score change"""
        # Premium adjustment factor based on risk score change
        if score_change < -0.2:
            adjustment_factor = 0.8  # 20% reduction
        elif score_change < -0.1:
            adjustment_factor = 0.9  # 10% reduction
        elif score_change < 0:
            adjustment_factor = 0.95  # 5% reduction
        elif score_change > 0.2:
            adjustment_factor = 1.3  # 30% increase
        elif score_change > 0.1:
            adjustment_factor = 1.15  # 15% increase
        else:
            adjustment_factor = 1.05  # 5% increase
        
        return original_premium * adjustment_factor
    
    def _get_risk_category(self, risk_score: float) -> str:
        """Get risk category based on score"""
        if risk_score < 0.3:
            return "Low Risk"
        elif risk_score < 0.6:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def _get_impact_magnitude(self, score_change: float) -> str:
        """Get impact magnitude description"""
        if score_change > 0.2:
            return "Significant"
        elif score_change > 0.1:
            return "Moderate"
        elif score_change > 0.05:
            return "Minor"
        else:
            return "Minimal"
    
    def _generate_simulation_recommendations(self, scenario_changes: dict, impact_metrics: dict) -> list:
        """Generate recommendations based on simulation results"""
        recommendations = []
        
        if impact_metrics['score_improvement']:
            recommendations.append("✅ Risk improvement achieved - consider implementing these changes")
            
            if 'add_telematics' in str(scenario_changes):
                recommendations.append("📱 Telematics program shows strong risk reduction")
            
            if 'defensive_driving' in str(scenario_changes):
                recommendations.append("🚗 Defensive driving course provides significant benefits")
            
            if 'vehicle_upgrade' in str(scenario_changes):
                recommendations.append("🚙 Vehicle upgrade to Toyota shows positive impact")
            
            if impact_metrics['premium_change_percentage'] < -5:
                recommendations.append("💰 Potential for premium reduction with these changes")
        else:
            recommendations.append("⚠️ Risk score increased - consider alternative mitigation strategies")
        
        if impact_metrics['risk_category_change'] != "No Change":
            recommendations.append(f"📊 Risk category changed: {impact_metrics['risk_category_change']}")
        
        return recommendations
    
    def _calculate_confidence_level(self, scenario_changes: dict) -> float:
        """Calculate confidence level in simulation results"""
        # Higher confidence for more realistic scenarios
        confidence = 0.8  # Base confidence
        
        # Adjust based on scenario complexity
        total_changes = sum(len(changes) for changes in scenario_changes.values())
        if total_changes > 3:
            confidence -= 0.1  # More complex scenarios have lower confidence
        
        # Adjust based on scenario types
        if 'behavioral_changes' in scenario_changes:
            confidence -= 0.05  # Behavioral changes are harder to predict
        
        return max(0.5, min(0.95, confidence))
    
    def _generate_simulation_id(self) -> str:
        """Generate unique simulation ID"""
        import uuid
        return f"sim_{uuid.uuid4().hex[:8]}"
    
    def _get_original_risk_score(self, base_data: dict) -> float:
        """Get original risk score from base data"""
        return base_data.get('risk_score', 0.5)
    
    def run_what_if_analysis(self, base_data: dict, scenarios: list) -> dict:
        """Run multiple what-if scenarios and compare results"""
        results = {}
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'Scenario_{i+1}')
            scenario_changes = scenario.get('changes', {})
            
            result = self.simulate_impact(base_data, scenario_changes)
            results[scenario_name] = result
        
        # Compare scenarios
        comparison = self._compare_scenarios(results)
        
        return {
            'scenario_results': results,
            'comparison': comparison,
            'best_scenario': self._identify_best_scenario(results),
            'recommendations': self._generate_comparison_recommendations(results)
        }
    
    def _compare_scenarios(self, results: dict) -> dict:
        """Compare multiple scenarios"""
        comparisons = {
            'risk_improvement': {},
            'premium_savings': {},
            'confidence_levels': {}
        }
        
        for scenario_name, result in results.items():
            comparisons['risk_improvement'][scenario_name] = result['risk_score_change']
            comparisons['premium_savings'][scenario_name] = result['impact_metrics']['premium_change']
            comparisons['confidence_levels'][scenario_name] = result['confidence_level']
        
        return comparisons
    
    def _identify_best_scenario(self, results: dict) -> str:
        """Identify the best scenario based on risk improvement and premium savings"""
        best_scenario = "No scenarios available"
        best_score = float('-inf')
        
        for scenario_name, result in results.items():
            # Score based on risk improvement and premium savings
            risk_improvement = -result['risk_score_change']  # Negative because lower is better
            premium_savings = -result['impact_metrics']['premium_change']  # Negative because lower premium is better
            confidence = result['confidence_level']
            
            # Weighted score
            score = (risk_improvement * 0.5) + (premium_savings * 0.3) + (confidence * 0.2)
            
            if score > best_score:
                best_score = score
                best_scenario = scenario_name
        
        return best_scenario
    
    def _generate_comparison_recommendations(self, results: dict) -> list:
        """Generate recommendations based on scenario comparison"""
        recommendations = []
        
        # Find scenarios with risk improvement
        improving_scenarios = [
            name for name, result in results.items() 
            if result['risk_score_change'] < 0
        ]
        
        if improving_scenarios:
            recommendations.append(f"✅ {len(improving_scenarios)} scenarios show risk improvement")
            
            # Find scenario with best premium savings
            best_premium_savings = min(
                results.items(),
                key=lambda x: x[1]['impact_metrics']['premium_change']
            )
            recommendations.append(f"💰 Best premium savings: {best_premium_savings[0]} (${best_premium_savings[1]['impact_metrics']['premium_change']:.2f})")
        
        # Find scenario with highest confidence
        highest_confidence = max(
            results.items(),
            key=lambda x: x[1]['confidence_level']
        )
        recommendations.append(f"🎯 Highest confidence: {highest_confidence[0]} ({highest_confidence[1]['confidence_level']:.1%})")
        
        return recommendations

      