import os

# URL of the MCP search server.
# Defaults to the Docker-internal hostname used in docker-compose.
# Override via MCP_SERVER_URL environment variable when running outside
# the default finagent_net Docker network (e.g., local dev, CI).
MCP_SERVER_URL: str = os.environ.get("MCP_SERVER_URL", "http://mcp_server:8000")
