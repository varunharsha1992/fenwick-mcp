"""Shared helper for every tool: one log line format, one error shape, one validator, one way to run a tool.

Used by the local Python tools, by the MCP server, and by the loop. Standard library only.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

TOOL_TIMEOUT_S = 10.0          # every tool call is bounded, local or remote
LOG_TEXT_LIMIT = 300           # logs carry a preview, never a full payload
SECRET_KEYS = re.compile(r"token|secret|password|api_?key|authorization", re.I)
JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}

# Logs go to STDERR. In the MCP server stdout is the protocol stream, so this is a rule, not a preference.
logger = logging.getLogger("agent")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log(event: str, **fields: Any) -> dict:
    """One structured log line (JSON). Returns the record so callers can also keep it in a trace."""
    record = {"event": event, **fields}
    logger.info(json.dumps(record, ensure_ascii=False, default=str))
    return record


def truncate(value: Any, limit: int = LOG_TEXT_LIMIT) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"


def redact(args: dict) -> dict:
    """Arguments as they may appear in a log: secrets masked, long values cut."""
    return {k: "***" if SECRET_KEYS.search(k) else truncate(v, 120) for k, v in args.items()}


class ToolError(Exception):
    """A failure the model is allowed to see: a code, a message, and whether trying again could help."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


def ok(data: Any) -> dict:
    return {"ok": True, "data": data}


def error(code: str, message: str, retryable: bool = False) -> dict:
    return {"ok": False, "error": {"code": code, "message": message, "retryable": retryable}}


@dataclass
class Tool:
    """A tool = what the model reads (name, description, parameters) + what our code runs (fn)."""
    name: str
    description: str
    parameters: dict                 # JSON Schema for the arguments
    fn: Callable[..., Any]
    needs_filer: bool = False        # fn also takes filed_by=..., supplied by the loop, never by the model
    side_effect: bool = False        # fn changes something outside the conversation
    remote: bool = False             # fn(args, meta) is an async call to another process (MCP)
    server: str = "local"            # where the tool lives: local, fenwick, github, mslearn
    hosted: bool = False             # somebody else runs it: keep its full result so a replay needs no network

    def schema(self) -> dict:
        """The part of the tool that is sent to the model."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def validate(args: Any, schema: dict) -> None:
    """Check arguments against the tool's JSON Schema before any work happens."""
    if not isinstance(args, dict):
        raise ToolError("invalid_arguments", "arguments must be a JSON object")
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            raise ToolError("invalid_arguments", f"missing required argument '{name}'")
    for name, value in args.items():
        if name not in props:
            raise ToolError("invalid_arguments", f"unknown argument '{name}'; allowed: {sorted(props)}")
        kind = props[name].get("type", "")
        expected = JSON_TYPES.get(kind) if isinstance(kind, str) else None      # check only what we understand
        is_bool = isinstance(value, bool)                 # in Python True is also an int: don't let it pass as one
        if expected and (not isinstance(value, expected) or is_bool != (expected is bool)):
            raise ToolError("invalid_arguments", f"'{name}' must be of type {props[name]['type']}")
        if "enum" in props[name] and value not in props[name]["enum"]:
            raise ToolError("invalid_arguments", f"'{name}' must be one of {props[name]['enum']}")


def execute(tool: Tool, args: dict, filed_by: str | None = None) -> dict:
    """Validate, run a local tool, and turn any failure into the structured error shape. Never raises."""
    try:
        validate(args, tool.parameters)
        extra = {"filed_by": filed_by} if tool.needs_filer else {}
        return ok(tool.fn(**args, **extra))
    except ToolError as e:
        return error(e.code, e.message, e.retryable)
    except Exception as e:                       # a bug or an outage inside the tool: report it, don't crash
        return error("tool_failed", f"{type(e).__name__}: {e}")


async def call_tool(tool: Tool | None, name: str, args: dict, *, correlation_id: str, step_id: int,
                    filed_by: str | None = None, timeout_s: float = TOOL_TIMEOUT_S) -> tuple[dict, dict]:
    """How the loop runs ONE tool call: bounded by a timeout, always logged. Returns (result, log record)."""
    started = time.perf_counter()
    if tool is None:
        result = error("unknown_tool", f"there is no tool called '{name}'")
    else:
        try:
            async with asyncio.timeout(timeout_s):
                if tool.remote:
                    validate(args, tool.parameters)
                    result = await tool.fn(args, {"correlation_id": correlation_id, "step_id": step_id})
                else:
                    result = await asyncio.to_thread(execute, tool, args, filed_by)
        except ToolError as e:
            result = error(e.code, e.message, e.retryable)
        except TimeoutError:
            result = error("timeout", f"'{name}' did not answer within {timeout_s:g}s", retryable=True)
        except Exception as e:
            result = error("tool_failed", f"{type(e).__name__}: {e}")
    record = log("tool_call", correlation_id=correlation_id, step_id=step_id, tool=name,
                 server=tool.server if tool else None, args=redact(args),
                 duration_ms=round((time.perf_counter() - started) * 1000), ok=result["ok"],
                 error=None if result["ok"] else result["error"]["code"], output=truncate(result))
    if tool is not None and tool.hosted:
        record["result"] = result            # in the trace, not in the log line
    return result, record

