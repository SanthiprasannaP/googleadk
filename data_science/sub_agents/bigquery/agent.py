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

"""Database Agent: get data from database (BigQuery) using NL2SQL.

This module defines the BigQuery database agent that handles natural language to SQL
conversion and query execution. It provides an interface for querying insurance
data stored in BigQuery using conversational language.

The agent supports multiple NL2SQL methods:
- BASELINE: Standard natural language to SQL conversion
- CHASE: Advanced SQL generation with schema-aware reasoning

Key Features:
- Natural language to SQL conversion
- Query validation and error handling
- Schema-aware query generation
- Support for multiple NL2SQL methodologies

Dependencies:
- Google ADK for agent framework
- BigQuery client for data access
- ChaseSQL tools for advanced query generation
"""

import os

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from . import tools
from .chase_sql import chase_db_tools
from .prompts import return_instructions_bigquery

# Determine which NL2SQL method to use (defaults to BASELINE)
NL2SQL_METHOD = os.getenv("NL2SQL_METHOD", "BASELINE")


def setup_before_agent_call(callback_context: CallbackContext) -> None:
    """Setup the BigQuery agent before each call.
    
    This function initializes the database settings and configuration for the
    BigQuery agent. It ensures that the agent has access to the necessary
    database connection parameters and schema information.
    
    Args:
        callback_context: The callback context containing session state and
                         agent configuration.
    
    Side Effects:
        - Updates callback_context.state with database settings
        - Initializes database configuration if not present
    
    Note:
        This function is called automatically before each agent invocation
        to ensure proper database setup.
    """
    # Initialize database settings if not present in session state
    if "database_settings" not in callback_context.state:
        callback_context.state["database_settings"] = \
            tools.get_database_settings()


# BigQuery database agent for natural language to SQL conversion
database_agent = Agent(
    model=os.getenv("BIGQUERY_AGENT_MODEL"),
    name="database_agent",
    instruction=return_instructions_bigquery(),
    tools=[
        # Select the appropriate NL2SQL tool based on configuration
        (
            chase_db_tools.initial_bq_nl2sql
            if NL2SQL_METHOD == "CHASE"
            else tools.initial_bq_nl2sql
        ),
        tools.run_bigquery_validation,
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
