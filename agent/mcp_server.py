"""Our MCP server: the on-call rota and a SIMULATED inbox. Any MCP client can list these tools and call them.

The name, the arguments and the docstring of each function become the tool the model sees.
The agent never imports this file. FastMCP Cloud runs it, and the agent reaches it by URL.
"""
import sys

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError          # an error whose text the model may read

from agent.tools import all_emails, fence, rows

mcp = MCPServer("fenwick", log_level="ERROR")


@mcp.tool()
def get_oncall(service: str, ctx: Context) -> dict:
    """Who is on call for a service right now. Always call this; the rota changes constantly."""
    log(ctx, "get_oncall", service)
    people = {p["employee_id"]: p for p in rows("identity_directory.csv")}
    for row in rows("oncall_schedule.csv"):
        if row["service"] == service.lower():
            primary, secondary = people[row["primary_employee_id"]], people[row["secondary_employee_id"]]
            return {"primary": primary["name"], "primary_id": primary["employee_id"], "primary_email": primary["email"],
                    "secondary": secondary["name"], "as_of": row["as_of"]}
    raise ToolError(f"no rota for '{service}'")


@mcp.tool()
def get_thread(thread_id: str, ctx: Context) -> dict:
    """Read one email thread from the support inbox."""
    log(ctx, "get_thread", thread_id)
    for email in all_emails():
        if email["thread_id"] == thread_id:
            return {"from": email["from"], "received_at": email["received_at"], "subject": fence(email["subject"]), "body": fence(email["body"])}
    raise ToolError(f"no thread '{thread_id}'")


@mcp.tool()
def create_draft(to: str, subject: str, body: str, ctx: Context) -> dict:
    """Write an email draft for a person to review. It is NOT sent. Separate several addresses with commas."""
    log(ctx, "create_draft", to)
    return {"status": "draft, not sent", "to": to, "subject": subject, "body": body}


def log(ctx, tool, what):
    """One line per request, with the id the agent sent along. The agent logged the same id when it sent the call."""
    sent = ctx.request_context.meta or {}
    print(f"server got {tool}({what}) call_id={sent.get('call_id')}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    mcp.run("streamable-http")          # only if hosting is down: http://127.0.0.1:8000/mcp
