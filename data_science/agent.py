# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top level agent for data agent multi-agents.

This module defines the main orchestrator agent for the insurance risk analytics
multi-agent system. It coordinates between different specialized agents to provide
comprehensive insurance data analysis, risk assessment, and predictive modeling.

The agent follows a workflow where it:
1. Gets data from database (e.g., BigQuery) using NL2SQL
2. Uses NL2Py to perform further data analysis as needed
3. Coordinates with specialized sub-agents for specific tasks

Key Components:
- Root agent orchestrator with multi-agent architecture
- Database integration via BigQuery
- Risk assessment and analytics capabilities
- Machine learning model integration
- SHAP-based explainability features

Dependencies:
- Google ADK for agent framework
- BigQuery for data storage and querying
- Various sub-agents for specialized tasks
"""
import os
from datetime import date
from typing import Dict, Any

from google.genai import types

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import load_artifacts

from .sub_agents import bqml_agent
from .sub_agents import risk_agent
from .sub_agents.bigquery.tools import (
    get_database_settings as get_bq_database_settings,
)
from .prompts import return_instructions_root
from .tools import call_db_agent, call_ds_agent

# Current date for dynamic content in agent instructions
date_today = date.today()


def setup_before_agent_call(callback_context: CallbackContext) -> None:
    """Setup the agent before each call.
    
    This function initializes the agent's state and configuration before processing
    each request. It sets up database settings, schema information, and agent
    instructions based on the current session state.
    
    Args:
        callback_context: The callback context containing session state and
                         invocation context for the agent.
    
    Side Effects:
        - Updates callback_context.state with database settings
        - Modifies agent instruction with schema information
        - Initializes database configuration if not present
    """
    # Initialize database settings in session state if not present
    if "database_settings" not in callback_context.state:
        db_settings = dict()
        db_settings["use_database"] = "BigQuery"
        callback_context.state["all_db_settings"] = db_settings

    # Set up schema in instruction for BigQuery database
    if callback_context.state["all_db_settings"]["use_database"] == "BigQuery":
        # Get BigQuery database settings including schema
        callback_context.state["database_settings"] = get_bq_database_settings()
        schema = callback_context.state["database_settings"]["bq_ddl_schema"]

        # Update agent instruction with schema information
        callback_context._invocation_context.agent.instruction = (
            return_instructions_root()
            + f"""

    --------- The BigQuery schema of the relevant data with a few sample rows. ---------
    {schema}

    """
        )


# Main orchestrator agent for the insurance risk analytics system
root_agent = Agent(
    model=os.getenv("ROOT_AGENT_MODEL"),
    name="conductor_multiagent",
    instruction=return_instructions_root(),
    global_instruction=(
        f"""
        🏢 **SecureDrive Insurance - AI-Powered Risk Assessment System**
        
        Welcome to our advanced multi-agent data science platform designed specifically for insurance risk assessment and analytics. Today's date: {date_today}
        
        **🎯 What I Can Do:**
        
        **📊 Data Analytics & Insights:**
        - Query insurance databases (BigQuery) using natural language
        - Perform advanced data analysis and statistical modeling
        - Generate comprehensive reports and visualizations
        - Analyze customer segments and claims patterns
        
        **🤖 Machine Learning & AI:**
        - BigQuery ML model training and deployment
        - Risk scoring and predictive modeling
        - SHAP-based explainability for transparent AI decisions
        - Fraud detection and anomaly identification
        
        **📈 Risk Assessment Services:**
        - Individual policy risk scoring with confidence intervals
        - Explanability of the risk scoring
        - What-if scenario analysis and impact simulation
        - Customer segmentation and risk profiling
        - Claims processing optimization
        
        **📋 Available Tools:**
        - **Database Agent**: Natural language to SQL queries
        - **Analytics Agent**: Python-based data analysis
        - **BQML Agent**: BigQuery ML model operations
        - **Risk Agent**: Insurance risk assessment with SHAP explainability
        
        **💡 How to Use Me:**
        Simply ask questions in natural language about:
        - Insurance data analysis ("Show me claims trends by region")
        - Risk assessment ("Calculate risk score for this policy")
        - Machine learning ("Train a fraud detection model")
        - What-if scenarios ("What if we change the premium amount?")
        
        I'll automatically route your request to the appropriate specialized agent and provide comprehensive, actionable insights with full transparency.
        
        Ready to transform your insurance data into actionable intelligence! 🚀
        """
    ),
    sub_agents=[bqml_agent, risk_agent],
    tools=[
        call_db_agent,
        call_ds_agent,
        load_artifacts,
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
