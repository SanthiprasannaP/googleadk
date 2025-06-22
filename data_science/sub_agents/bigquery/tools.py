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

"""BigQuery tools for natural language to SQL conversion and query execution.

This module provides the core tools for the BigQuery database agent, enabling
natural language to SQL conversion and query validation. It handles schema
retrieval, SQL generation, and query execution against BigQuery.

Key Features:
- Natural language to SQL conversion using LLM models
- BigQuery schema retrieval and DDL generation
- Query validation and execution
- Database connection management
- ChaseSQL integration for advanced query generation

Core Functions:
- initial_bq_nl2sql: Convert natural language to SQL queries
- run_bigquery_validation: Validate and execute SQL queries
- get_bigquery_schema: Retrieve and format BigQuery schema
- get_database_settings: Manage database configuration

Dependencies:
- Google Cloud BigQuery client
- Google GenAI for LLM integration
- ChaseSQL constants and utilities
"""

import datetime
import logging
import os
import re
from typing import Optional, Dict, Any

from data_science.utils.utils import get_env_var
from google.adk.tools import ToolContext
from google.cloud import bigquery
from google.genai import Client

from .chase_sql import chase_constants

# Configuration constants
project = os.getenv("BQ_PROJECT_ID", None)  # BigQuery project ID
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")  # Google Cloud location
llm_client = Client(vertexai=True, project=project, location=location)  # LLM client for SQL generation

MAX_NUM_ROWS = 80  # Maximum number of rows to return in queries

# Global variables for caching
database_settings = None
bq_client = None


def get_bq_client() -> bigquery.Client:
    """Get or create a BigQuery client instance.
    
    This function implements a singleton pattern for the BigQuery client to
    avoid creating multiple connections. It uses the project ID from environment
    variables to initialize the client.
    
    Returns:
        bigquery.Client: Configured BigQuery client instance.
    
    Raises:
        ValueError: If BQ_PROJECT_ID environment variable is not set.
    
    Note:
        The client is cached globally to improve performance across multiple
        function calls.
    """
    global bq_client
    if bq_client is None:
        bq_client = bigquery.Client(project=get_env_var("BQ_PROJECT_ID"))
    return bq_client


def get_database_settings() -> Dict[str, Any]:
    """Get cached database settings or create new ones.
    
    This function provides access to database configuration including schema
    information and ChaseSQL constants. It caches the settings to avoid
    repeated schema retrieval.
    
    Returns:
        Dict containing database configuration:
            - bq_project_id: BigQuery project ID
            - bq_dataset_id: BigQuery dataset ID
            - bq_ddl_schema: Schema in DDL format with examples
            - chase_sql_constants: ChaseSQL-specific configuration
    
    Note:
        Settings are cached globally to improve performance. Use
        update_database_settings() to refresh the cache.
    """
    global database_settings
    if database_settings is None:
        database_settings = update_database_settings()
    return database_settings


def update_database_settings() -> Dict[str, Any]:
    """Update database settings by retrieving fresh schema information.
    
    This function refreshes the database configuration by retrieving the
    current BigQuery schema and combining it with ChaseSQL constants.
    It's useful when schema changes or when fresh data is needed.
    
    Returns:
        Dict containing updated database configuration with schema and constants.
    
    Side Effects:
        - Updates global database_settings variable
        - Retrieves fresh schema from BigQuery
    """
    global database_settings
    
    # Retrieve current BigQuery schema
    ddl_schema = get_bigquery_schema(
        get_env_var("BQ_DATASET_ID"),
        client=get_bq_client(),
        project_id=get_env_var("BQ_PROJECT_ID"),
    )
    
    # Combine schema with ChaseSQL constants
    database_settings = {
        "bq_project_id": get_env_var("BQ_PROJECT_ID"),
        "bq_dataset_id": get_env_var("BQ_DATASET_ID"),
        "bq_ddl_schema": ddl_schema,
        # Include ChaseSQL-specific constants for advanced query generation
        **chase_constants.chase_sql_constants_dict,
    }
    return database_settings


def get_bigquery_schema(dataset_id: str, client: Optional[bigquery.Client] = None, project_id: Optional[str] = None) -> str:
    """Retrieves schema and generates DDL with example values for a BigQuery dataset.

    This function connects to BigQuery, retrieves the schema for all tables in
    the specified dataset, and generates DDL statements with example data. It's
    used to provide context for natural language to SQL conversion.

    Args:
        dataset_id: The ID of the BigQuery dataset (e.g., 'my_dataset').
        client: Optional BigQuery client. If None, creates a new client.
        project_id: Optional project ID. Used if client is None.

    Returns:
        str: A string containing the generated DDL statements with example data.

    Example:
        >>> schema = get_bigquery_schema("insurance_data")
        >>> print(schema)
        CREATE OR REPLACE TABLE `project.dataset.customers` (
          `customer_id` STRING,
          `age` INT64,
          `income` FLOAT64,
        );
        
        -- Example values for table `project.dataset.customers`:
        INSERT INTO `project.dataset.customers` VALUES
        ('C001', 35, 75000);
    """
    if client is None:
        client = bigquery.Client(project=project_id)

    # Create dataset reference
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    ddl_statements = ""

    # Iterate through all tables in the dataset
    for table in client.list_tables(dataset_ref):
        table_ref = dataset_ref.table(table.table_id)
        table_obj = client.get_table(table_ref)

        # Skip views, only process tables
        if table_obj.table_type != "TABLE":
            continue

        # Generate DDL statement for the table
        ddl_statement = f"CREATE OR REPLACE TABLE `{table_ref}` (\n"

        # Add each field to the DDL
        for field in table_obj.schema:
            ddl_statement += f"  `{field.name}` {field.field_type}"
            if field.mode == "REPEATED":
                ddl_statement += " ARRAY"
            if field.description:
                ddl_statement += f" COMMENT '{field.description}'"
            ddl_statement += ",\n"

        # Remove trailing comma and close the statement
        ddl_statement = ddl_statement[:-2] + "\n);\n\n"

        # Add example values if available (limited to first 5 rows)
        rows = client.list_rows(table_ref, max_results=5).to_dataframe()
        if not rows.empty:
            ddl_statement += f"-- Example values for table `{table_ref}`:\n"
            for _, row in rows.iterrows():
                ddl_statement += f"INSERT INTO `{table_ref}` VALUES\n"
                example_row_str = "("
                
                # Format each value appropriately
                for value in row.values:
                    if isinstance(value, str):
                        example_row_str += f"'{value}',"
                    elif value is None:
                        example_row_str += "NULL,"
                    else:
                        example_row_str += f"{value},"
                
                # Remove trailing comma and close the statement
                example_row_str = example_row_str[:-1] + ");\n\n"
                ddl_statement += example_row_str

        ddl_statements += ddl_statement

    return ddl_statements


def initial_bq_nl2sql(
    question: str,
    tool_context: ToolContext,
) -> str:
    """Generates an initial SQL query from a natural language question.

    This function uses a large language model to convert natural language
    questions into BigQuery SQL queries. It provides the model with schema
    information and guidelines to generate accurate, syntactically correct SQL.

    Args:
        question: Natural language question to be converted to SQL.
        tool_context: Tool context containing database settings and schema.

    Returns:
        str: Generated SQL statement to answer the question.

    Note:
        The function uses a prompt template that includes:
        - Schema information with example data
        - SQL generation guidelines
        - BigQuery-specific syntax requirements
        - Row limit constraints
    """
    # Prompt template for SQL generation
    prompt_template = """
You are a BigQuery SQL expert tasked with answering user's questions about BigQuery tables by generating SQL queries in the GoogleSql dialect.  Your task is to write a Bigquery SQL query that answers the following question while using the provided context.

**Guidelines:**

- **Table Referencing:** Always use the full table name with the database prefix in the SQL statement.  Tables should be referred to using a fully qualified name with enclosed in backticks (`) e.g. `project_name.dataset_name.table_name`.  Table names are case sensitive.
- **Joins:** Join as few tables as possible. When joining tables, ensure all join columns are the same data type. Analyze the database and the table schema provided to understand the relationships between columns and tables.
- **Aggregations:**  Use all non-aggregated columns from the `SELECT` statement in the `GROUP BY` clause.
- **SQL Syntax:** Return syntactically and semantically correct SQL for BigQuery with proper relation mapping (i.e., project_id, owner, table, and column relation). Use SQL `AS` statement to assign a new name temporarily to a table column or even a table wherever needed. Always enclose subqueries and union queries in parentheses.
- **Column Usage:** Use *ONLY* the column names (column_name) mentioned in the Table Schema. Do *NOT* use any other column names. Associate `column_name` mentioned in the Table Schema only to the `table_name` specified under Table Schema.
- **FILTERS:** You should write query effectively  to reduce and minimize the total rows to be returned. For example, you can use filters (like `WHERE`, `HAVING`, etc. (like 'COUNT', 'SUM', etc.) in the SQL query.
- **LIMIT ROWS:**  The maximum number of rows returned should be less than {MAX_NUM_ROWS}.

**Schema:**

The database structure is defined by the following table schemas (possibly with sample rows):

```
{SCHEMA}
```

**Natural language question:**

```
{QUESTION}
```

**Think Step-by-Step:** Carefully consider the schema, question, guidelines, and best practices outlined above to generate the correct BigQuery SQL.

   """

    # Get schema from tool context
    ddl_schema = tool_context.state["database_settings"]["bq_ddl_schema"]

    # Format the prompt with schema and question
    prompt = prompt_template.format(
        MAX_NUM_ROWS=MAX_NUM_ROWS, SCHEMA=ddl_schema, QUESTION=question
    )

    # Generate SQL using the LLM
    response = llm_client.models.generate_content(
        model=os.getenv("BASELINE_NL2SQL_MODEL"),
        contents=prompt,
        config={"temperature": 0.1},  # Low temperature for consistent SQL generation
    )

    # Extract and clean the SQL response
    sql = response.text
    if sql:
        sql = sql.replace("```sql", "").replace("```", "").strip()
    
    return sql


def run_bigquery_validation(
    sql_string: str,
    tool_context: ToolContext,
) -> str:
    """Validates BigQuery SQL syntax and functionality.

    This function validates the provided SQL string by attempting to execute it
    against BigQuery in dry-run mode. It performs the following checks:

    1. **SQL Cleanup:**  Preprocesses the SQL string using a `cleanup_sql`
    function
    2. **DML/DDL Restriction:**  Rejects any SQL queries containing DML or DDL
       statements (e.g., UPDATE, DELETE, INSERT, CREATE, ALTER) to ensure
       read-only operations.
    3. **Syntax and Execution:** Sends the cleaned SQL to BigQuery for validation.
       If the query is syntactically correct and executable, it retrieves the
       results.
    4. **Result Analysis:**  Checks if the query produced any results. If so, it
       formats the first few rows of the result set for inspection.

    Args:
        sql_string (str): The SQL query string to validate.
        tool_context (ToolContext): The tool context to use for validation.

    Returns:
        str: A message indicating the validation outcome. This includes:
             - "Valid SQL. Results: ..." if the query is valid and returns data.
             - "Valid SQL. Query executed successfully (no results)." if the query
                is valid but returns no data.
             - "Invalid SQL: ..." if the query is invalid, along with the error
                message from BigQuery.
    """

    def cleanup_sql(sql_string):
        """Processes the SQL string to get a printable, valid SQL string."""

        # 1. Remove backslashes escaping double quotes
        sql_string = sql_string.replace('\\"', '"')

        # 2. Remove backslashes before newlines (the key fix for this issue)
        sql_string = sql_string.replace("\\\n", "\n")  # Corrected regex

        # 3. Replace escaped single quotes
        sql_string = sql_string.replace("\\'", "'")

        # 4. Replace escaped newlines (those not preceded by a backslash)
        sql_string = sql_string.replace("\\n", "\n")

        # 5. Add limit clause if not present
        if "limit" not in sql_string.lower():
            sql_string = sql_string + " limit " + str(MAX_NUM_ROWS)

        return sql_string

    logging.info("Validating SQL: %s", sql_string)
    sql_string = cleanup_sql(sql_string)
    logging.info("Validating SQL (after cleanup): %s", sql_string)

    final_result = {"query_result": None, "error_message": None}

    # More restrictive check for BigQuery - disallow DML and DDL
    if re.search(
        r"(?i)(update|delete|drop|insert|create|alter|truncate|merge)", sql_string
    ):
        final_result["error_message"] = (
            "Invalid SQL: Contains disallowed DML/DDL operations."
        )
        return final_result

    try:
        query_job = get_bq_client().query(sql_string)
        results = query_job.result()  # Get the query results

        if results.schema:  # Check if query returned data
            rows = [
                {
                    key: (
                        value
                        if not isinstance(value, datetime.date)
                        else value.strftime("%Y-%m-%d")
                    )
                    for (key, value) in row.items()
                }
                for row in results
            ][
                :MAX_NUM_ROWS
            ]  # Convert BigQuery RowIterator to list of dicts
            # return f"Valid SQL. Results: {rows}"
            final_result["query_result"] = rows

            tool_context.state["query_result"] = rows

        else:
            final_result["error_message"] = (
                "Valid SQL. Query executed successfully (no results)."
            )

    except (
        Exception
    ) as e:  # Catch generic exceptions from BigQuery  # pylint: disable=broad-exception-caught
        final_result["error_message"] = f"Invalid SQL: {e}"

    print("\n run_bigquery_validation final_result: \n", final_result)

    return final_result
