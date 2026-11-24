"""The MCP server: Fenwick's on-call scheduler and a SIMULATED inbox, over the Model Context Protocol.

Transport: stdio. The client starts this file as a subprocess and talks to it over stdin/stdout.
That is the right fit for a local teaching demo: no port, no auth, nothing to deploy.
Because stdout IS the protocol stream, every log line goes to stderr (agent.toolkit.logger does that).

    python -m agent.mcp_server          started for you by agent.mcp_client.connect_mcp()

Each tool body is the same Python function a local tool would use (agent/tools.py), run through
the same validate-and-structured-error helper (agent.toolkit.execute).
"""
import json
import time
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import Field

from agent.toolkit import Tool, execute, log, redact, truncate
from agent.tools import CREATE_DRAFT, GET_ONCALL, GET_THREAD, SEARCH_THREADS

mcp = MCPServer("fenwick", log_level="WARNING")


def serve(tool: Tool, args: dict, ctx: Context) -> str:
    """Run one tool for one MCP request and log it. A bad call becomes a structured error; the server stays up."""
    started = time.perf_counter()
    meta = ctx.request_context.meta or {}                 # the client sends the run's ids along with the call
    result = execute(tool, args)
    log("mcp_request", request_id=ctx.request_id, correlation_id=meta.get("correlation_id"),
        step_id=meta.get("step_id"), tool=tool.name, args=redact(args),
        duration_ms=round((time.perf_counter() - started) * 1000), ok=result["ok"],
        error=None if result["ok"] else result["error"]["code"], output=truncate(result))
    return json.dumps(result, ensure_ascii=False)


def says(tool: Tool, arg: str) -> Field:
    """The argument description the model reads, taken from the one place it is written."""
    return Field(description=tool.parameters["properties"][arg]["description"])


@mcp.tool(name=GET_ONCALL.name, description=GET_ONCALL.description)
def get_oncall(service: Annotated[str, says(GET_ONCALL, "service")], ctx: Context) -> str:
    return serve(GET_ONCALL, {"service": service}, ctx)


@mcp.tool(name=SEARCH_THREADS.name, description=SEARCH_THREADS.description)
def search_threads(query: Annotated[str, says(SEARCH_THREADS, "query")], ctx: Context) -> str:
    return serve(SEARCH_THREADS, {"query": query}, ctx)


@mcp.tool(name=GET_THREAD.name, description=GET_THREAD.description)
def get_thread(thread_id: Annotated[str, says(GET_THREAD, "thread_id")], ctx: Context) -> str:
    return serve(GET_THREAD, {"thread_id": thread_id}, ctx)


@mcp.tool(name=CREATE_DRAFT.name, description=CREATE_DRAFT.description)
def create_draft(to: Annotated[str, says(CREATE_DRAFT, "to")], subject: Annotated[str, says(CREATE_DRAFT, "subject")],
                 body: Annotated[str, says(CREATE_DRAFT, "body")], ctx: Context) -> str:
    return serve(CREATE_DRAFT, {"to": to, "subject": subject, "body": body}, ctx)


if __name__ == "__main__":
    mcp.run()          # stdio
