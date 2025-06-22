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

"""Top level tools for data agent multi-agents.

This module provides the main orchestration tools that coordinate between different
specialized agents in the insurance risk analytics system. It handles the workflow
of routing requests to appropriate sub-agents and managing data flow between them.

The module implements a two-stage workflow:
1. Database Agent: Handles natural language to SQL conversion and data retrieval
2. Data Science Agent: Performs advanced analytics and statistical modeling

Key Functions:
- call_db_agent: Routes database queries to the BigQuery agent
- call_ds_agent: Routes analytics requests to the data science agent

Dependencies:
- Google ADK tools for agent communication
- Sub-agents for specialized tasks (db_agent, ds_agent)
"""

from typing import Any, Dict
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .sub_agents import ds_agent, db_agent


async def call_db_agent(
    question: str,
    tool_context: ToolContext,
) -> str:
    """Tool to call database (nl2sql) agent.
    
    This function routes natural language questions to the database agent for
    conversion to SQL queries and execution against BigQuery. It handles the
    first stage of the analytics workflow by retrieving data from the database.
    
    Args:
        question: Natural language question to be converted to SQL and executed.
        tool_context: Context containing session state and database settings.
    
    Returns:
        str: The result of the database query execution, typically containing
             data in a structured format or error messages.
    
    Side Effects:
        - Updates tool_context.state["db_agent_output"] with the query result
        - Logs the database being used for debugging purposes
    
    Example:
        >>> result = await call_db_agent(
        ...     "Show me claims data for the last 6 months",
        ...     tool_context
        ... )
    """
    # Log which database is being used for debugging
    print(
        "\n call_db_agent.use_database:"
        f' {tool_context.state["all_db_settings"]["use_database"]}'
    )

    # Create agent tool wrapper for the database agent
    agent_tool = AgentTool(agent=db_agent)

    # Execute the database query asynchronously
    db_agent_output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    
    # Store the result in session state for potential use by other agents
    tool_context.state["db_agent_output"] = db_agent_output
    return db_agent_output


async def call_ds_agent(
    question: str,
    tool_context: ToolContext,
) -> str:
    """Tool to call data science (nl2py) agent.

    This function routes analytics questions to the data science agent for
    advanced data analysis and statistical modeling. It handles the second stage
    of the analytics workflow by processing data retrieved from the database.
    
    If the question is "N/A", it returns the previous database agent output
    without performing additional analysis.
    
    Args:
        question: Natural language question for data analysis, or "N/A" to skip.
        tool_context: Context containing session state and previous query results.
    
    Returns:
        str: The result of the data science analysis, typically containing
             insights, visualizations, or statistical summaries.
    
    Side Effects:
        - Updates tool_context.state["ds_agent_output"] with the analysis result
        - Accesses previous database query results from session state
    
    Example:
        >>> result = await call_ds_agent(
        ...     "Analyze the correlation between age and claims frequency",
        ...     tool_context
        ... )
    """
    # If no analysis is requested, return the database output directly
    if question == "N/A":
        return tool_context.state["db_agent_output"]

    # Get the data from previous database query for analysis
    input_data = tool_context.state["query_result"]

    # Combine the analysis question with the available data
    question_with_data = f"""
  Question to answer: {question}

  Actual data to analyze prevoius quesiton is already in the following:
  {input_data}

  """

    # Create agent tool wrapper for the data science agent
    agent_tool = AgentTool(agent=ds_agent)

    # Execute the data science analysis asynchronously
    ds_agent_output = await agent_tool.run_async(
        args={"request": question_with_data}, tool_context=tool_context
    )
    
    # Store the result in session state for potential use by other components
    tool_context.state["ds_agent_output"] = ds_agent_output
    return ds_agent_output
