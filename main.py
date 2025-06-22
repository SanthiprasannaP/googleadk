"""Main entry point for the Insurance Risk Analytics Multi-Agent System.

This module serves as the primary entry point for the FastAPI-based web application
that provides insurance risk assessment, analytics, and predictive modeling services.
It configures the web server, sets up CORS, and initializes the multi-agent system.

The application provides:
- REST API endpoints for insurance risk analysis
- Web interface for interactive risk assessment
- Multi-agent orchestration for complex analytics workflows
- Session management for user interactions

Key Features:
- FastAPI-based REST API
- Web interface for risk assessment
- Session persistence with SQLite
- CORS configuration for cross-origin requests
- Cloud Run deployment ready

Dependencies:
- FastAPI for web framework
- Uvicorn for ASGI server
- Google ADK for agent framework
- SQLite for session storage
"""

import os
from typing import Optional

import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

# Configuration constants
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory containing main.py
SESSION_DB_URL = "sqlite:///./sessions.db"  # SQLite database for session storage
ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]  # CORS allowed origins
SERVE_WEB_INTERFACE = True  # Enable web interface

# Initialize the FastAPI application with multi-agent system
app = get_fast_api_app(
    agents_dir=AGENT_DIR,  # Directory containing agent modules
    session_service_uri=SESSION_DB_URL,  # Session storage configuration
    allow_origins=ALLOWED_ORIGINS,  # CORS configuration
    web=SERVE_WEB_INTERFACE,  # Enable web interface
)

# Additional FastAPI routes can be added here if needed
# Example:
# @app.get("/hello")
# async def read_root():
#     return {"Hello": "World"}


def main() -> None:
    """Main function to start the FastAPI server.
    
    This function starts the Uvicorn ASGI server with the configured FastAPI
    application. It uses environment variables for port configuration to support
    deployment on platforms like Google Cloud Run.
    
    Environment Variables:
        PORT: Port number for the server (defaults to 8080)
    
    Side Effects:
        - Starts the web server
        - Binds to all interfaces (0.0.0.0)
        - Uses the specified port from environment or default
    """
    # Get port from environment variable (for Cloud Run compatibility) or use default
    port = int(os.environ.get("PORT", 8080))
    
    # Start the server with uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Bind to all interfaces
        port=port
    )


if __name__ == "__main__":
    main()