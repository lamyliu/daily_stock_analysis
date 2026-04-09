#!/usr/bin/env python3
"""
Claude CLI → OpenAI 兼容 API 适配器

将 claude -p 包装为 OpenAI 格式的 HTTP API，让任何支持 OpenAI SDK 的项目
直接使用 Claude 订阅，无需 API Key。

支持：chat completions、streaming、function calling (tool use)

启动: python scripts/claude-cli-adapter.py
访问: http://127.0.0.1:8877/v1/chat/completions
文档: http://127.0.0.1:8877/docs
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
import logging
from typing import AsyncGenerator, Optional, List, Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 8877
MAX_CONCURRENT = 3
TIMEOUT_SECONDS = 300

AVAILABLE_MODELS = [
    {"id": "claude-sonnet-4-5", "owned_by": "anthropic"},
    {"id": "claude-sonnet-4-6", "owned_by": "anthropic"},
    {"id": "claude-opus-4-6", "owned_by": "anthropic"},
    {"id": "claude-haiku-4-5", "owned_by": "anthropic"},
    # OpenAI-compatible names (mapped to Claude models)
    {"id": "gpt-5.4", "owned_by": "openai"},
    {"id": "gpt-5.4-mini", "owned_by": "openai"},
    {"id": "gpt-4o", "owned_by": "openai"},
    {"id": "gpt-4o-mini", "owned_by": "openai"},
]

MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5",
    # OpenAI model names → Claude
    "gpt-5.4": "claude-sonnet-4-6",
    "gpt-5.4-mini": "claude-haiku-4-5",
    "gpt-4o": "claude-sonnet-4-5",
    "gpt-4o-mini": "claude-haiku-4-5",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("claude-cli-adapter")

# ── Tool Use Support ─────────────────────────────────────────────────────────

TOOL_USE_SYSTEM_SUFFIX = """

## Tool Use Protocol

You have access to the following tools. When you need to call a tool, respond with ONLY a JSON object in this exact format (no other text before or after):

{"tool_calls": [{"name": "function_name", "arguments": {"arg1": "value1"}}]}

When you do NOT need to call a tool, respond normally with text content.

CRITICAL RULES:
- If you want to call tools, output ONLY the JSON object above, nothing else
- You can call multiple tools at once by adding more items to the tool_calls array
- After receiving tool results, synthesize a final answer for the user

### Available Tools:
"""

def format_tools_for_prompt(tools: List[Dict]) -> str:
    """Convert OpenAI tools format to prompt text."""
    if not tools:
        return ""
    lines = [TOOL_USE_SYSTEM_SUFFIX]
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            lines.append(f"\n**{fn['name']}**: {fn.get('description', '')}")
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])
            if props:
                lines.append("Parameters:")
                for pname, pinfo in props.items():
                    req = " (required)" if pname in required else ""
                    lines.append(f"  - {pname}: {pinfo.get('type', 'string')} — {pinfo.get('description', '')}{req}")
    return "\n".join(lines)

def parse_tool_calls(content: str) -> Optional[List[Dict]]:
    """Try to parse tool_calls from LLM response."""
    content = content.strip()
    # Try direct JSON parse
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "tool_calls" in data:
            return data["tool_calls"]
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown fence
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', content)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            if isinstance(data, dict) and "tool_calls" in data:
                return data["tool_calls"]
        except json.JSONDecodeError:
            pass
    # Try finding JSON object with tool_calls
    tc_match = re.search(r'\{"tool_calls":\s*\[[\s\S]*?\]\s*\}', content)
    if tc_match:
        try:
            data = json.loads(tc_match.group(0))
            return data.get("tool_calls")
        except json.JSONDecodeError:
            pass
    return None

# ── Core ──────────────────────────────────────────────────────────────────────

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)

JSON_ENFORCEMENT = "\n\nIMPORTANT: You MUST respond with raw JSON only. No markdown, no explanation, no code fences. Start your response with { and end with }. This is critical."

def build_cli_args(messages: List[Dict], model: str, stream: bool,
                   tools: Optional[List[Dict]] = None,
                   json_mode: bool = False) -> tuple:
    """Build claude CLI args and stdin prompt.

    Args:
        json_mode: When True, append a strict JSON-only enforcement suffix to the
                   system prompt.  Callers should set this when ``response_format``
                   is ``{"type": "json_object"}``.  When False the model is free to
                   respond in plain text (chat / conversational mode).
    """
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    has_tools = bool(tools)

    # Build system prompt with tool definitions or JSON enforcement
    effective_system = system or ""
    if has_tools:
        effective_system += format_tools_for_prompt(tools)
    elif json_mode:
        effective_system += JSON_ENFORCEMENT

    # Build conversation from messages
    conversation = []
    for m in messages:
        if m["role"] == "system":
            continue
        elif m["role"] == "user":
            conversation.append(f"Human: {m['content']}")
        elif m["role"] == "assistant":
            content = m.get("content", "")
            # Handle assistant messages with tool_calls (re-serialize)
            tool_calls = m.get("tool_calls")
            if tool_calls and not content:
                tc_json = json.dumps({"tool_calls": [
                    {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]}
                    for tc in tool_calls
                ]})
                conversation.append(f"Assistant: {tc_json}")
            elif content:
                conversation.append(f"Assistant: {content}")
        elif m["role"] == "tool":
            # Tool result — format as Human message with tool context
            tool_name = m.get("name", "tool")
            conversation.append(f"Human: [Tool Result for {tool_name}]: {m['content']}")

    # Merge system prompt into stdin (avoids ARG_MAX limit for long prompts)
    parts = []
    if effective_system:
        parts.append(f"[System Instructions]\n{effective_system}\n[End System Instructions]")
    parts.extend(conversation)
    prompt = "\n\n".join(parts)

    fmt = "stream-json" if stream else "json"
    cmd = ["claude", "-p", "--output-format", fmt, "--model", resolve_model(model),
           "--disable-slash-commands", "--no-session-persistence", "--tools", ""]

    return cmd, prompt

async def call_claude(messages: List[Dict], model: str,
                      tools: Optional[List[Dict]] = None,
                      json_mode: bool = False) -> dict:
    """Call claude -p and return parsed JSON result."""
    cmd, prompt = build_cli_args(messages, model, stream=False, tools=tools, json_mode=json_mode)
    log.info(f"calling: model={model}, tools={len(tools) if tools else 0}, prompt={len(prompt)} chars")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/tmp",
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode()),
            timeout=TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(504, f"claude CLI timed out after {TIMEOUT_SECONDS}s")

    out_raw = stdout.decode() if stdout else ""
    err_raw = stderr.decode().strip() if stderr else ""

    # Try to parse JSON even on non-zero exit (claude returns JSON with is_error)
    result = None
    try:
        result = json.loads(out_raw)
    except (json.JSONDecodeError, ValueError):
        pass

    if proc.returncode != 0:
        if result and result.get("is_error"):
            msg = result.get("result", "unknown error")
            log.error(f"claude CLI error: {msg}")
            raise HTTPException(502, f"claude error: {msg}")
        log.error(f"claude CLI failed (rc={proc.returncode}): stderr={err_raw} stdout={out_raw[:300]}")
        raise HTTPException(502, f"claude CLI error: {err_raw or out_raw[:300]}")

    if result is None:
        log.error(f"invalid JSON from claude: {out_raw[:500]}")
        raise HTTPException(502, "invalid JSON from claude CLI")

    if result.get("is_error"):
        raise HTTPException(502, f"claude error: {result.get('result', 'unknown')}")

    return result

def _clean_llm_content(text: str) -> str:
    """Clean LLM response: extract JSON from markdown fences if present."""
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if stripped.startswith('{') or stripped.startswith('['):
        return stripped
    json_start = text.find('{')
    json_end = text.rfind('}')
    if json_start >= 0 and json_end > json_start:
        return text[json_start:json_end + 1]
    return text

def to_openai_response(result: dict, model: str,
                       has_tools: bool = False) -> dict:
    """Convert claude JSON result to OpenAI chat completion format."""
    usage = result.get("usage", {})
    input_tokens = (usage.get("input_tokens", 0) +
                    usage.get("cache_creation_input_tokens", 0) +
                    usage.get("cache_read_input_tokens", 0))
    output_tokens = usage.get("output_tokens", 0)

    raw_content = result.get("result", "")
    content = _clean_llm_content(raw_content)

    # Check if response contains tool calls
    tool_calls_data = parse_tool_calls(content) if has_tools else None

    if tool_calls_data:
        # Return as tool_calls response
        openai_tool_calls = []
        for i, tc in enumerate(tool_calls_data):
            args = tc.get("arguments", {})
            openai_tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                },
            })
        log.info(f"tool_calls detected: {[tc['name'] for tc in tool_calls_data]}")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": openai_tool_calls,
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    # Regular text response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop" if result.get("stop_reason") == "end_turn" else "length",
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Claude CLI Adapter", version="2.0.0")

@app.get("/health")
async def health():
    return {"status": "ok", "max_concurrent": MAX_CONCURRENT, "tool_use": True}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": 0, "owned_by": m["owned_by"]}
            for m in AVAILABLE_MODELS
        ],
    }

@app.get("/api/tags")
async def ollama_tags():
    """Ollama-compatible model list."""
    return {
        "models": [
            {"name": m["id"], "model": m["id"], "modified_at": "2026-01-01T00:00:00Z", "size": 0}
            for m in AVAILABLE_MODELS
        ]
    }

@app.post("/api/chat")
async def ollama_chat(request: Request):
    """Ollama-compatible chat endpoint."""
    body = await request.json()
    model = body.get("model", "claude-sonnet-4-6")
    messages = body.get("messages", [])
    tools = body.get("tools")

    async with semaphore:
        result = await call_claude(messages, model, tools=tools)
        response = to_openai_response(result, model, has_tools=bool(tools))

        choice = response["choices"][0]
        ollama_resp = {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "message": choice["message"],
            "done": True,
            "total_duration": result.get("duration_ms", 0) * 1_000_000,
            "eval_count": response["usage"]["completion_tokens"],
            "prompt_eval_count": response["usage"]["prompt_tokens"],
        }
        return JSONResponse(content=ollama_resp)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completions with optional tool use."""
    body = await request.json()

    model = body.get("model", "claude-sonnet-4-5")
    messages = body.get("messages", [])
    tools = body.get("tools")
    stream = body.get("stream", False)
    response_format = body.get("response_format") or {}
    json_mode = response_format.get("type") == "json_object"

    async with semaphore:
        if stream and not tools:
            # Streaming without tools (tool use doesn't support streaming)
            return StreamingResponse(
                stream_claude_legacy(messages, model, json_mode=json_mode),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await call_claude(messages, model, tools=tools, json_mode=json_mode)
        response = to_openai_response(result, model, has_tools=bool(tools))

        cost = result.get("total_cost_usd", 0)
        duration = result.get("duration_ms", 0)
        tokens = response["usage"]["total_tokens"]
        finish = response["choices"][0].get("finish_reason", "?")
        log.info(f"done: {tokens} tokens, ${cost:.4f}, {duration}ms, finish={finish}")

        return JSONResponse(content=response)

async def stream_claude_legacy(messages: List[Dict], model: str, json_mode: bool = False) -> AsyncGenerator[str, None]:
    """Stream claude -p output as SSE (legacy, no tool use)."""
    cmd, prompt = build_cli_args(messages, model, stream=True, json_mode=json_mode)
    log.info(f"streaming: model={model}, prompt={len(prompt)} chars")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/tmp",
    )

    proc.stdin.write(prompt.encode())
    await proc.stdin.drain()
    proc.stdin.close()

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async for line in proc.stdout:
        text = line.decode().strip()
        if not text:
            continue
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "assistant" and "content" in event:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": event["content"]}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        elif event.get("type") == "result":
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"
    await proc.wait()

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    log.info(f"Claude CLI Adapter v2.0 starting on http://{HOST}:{PORT}")
    log.info(f"Models: {[m['id'] for m in AVAILABLE_MODELS]}")
    log.info(f"Max concurrent: {MAX_CONCURRENT}, Tool use: enabled")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
