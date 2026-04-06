#!/usr/bin/env python3
"""
Claude CLI → OpenAI 兼容 API 适配器

将 claude -p 包装为 OpenAI 格式的 HTTP API，让任何支持 OpenAI SDK 的项目
直接使用 Claude 订阅，无需 API Key。

启动: python scripts/claude-cli-adapter.py
访问: http://127.0.0.1:8877/v1/chat/completions
文档: http://127.0.0.1:8877/docs
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
import logging
from typing import AsyncGenerator, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
]

MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("claude-cli-adapter")

# ── Models ────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "claude-sonnet-4-5"
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False

# ── Core ──────────────────────────────────────────────────────────────────────

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)

JSON_ENFORCEMENT = "\n\nIMPORTANT: You MUST respond with raw JSON only. No markdown, no explanation, no code fences. Start your response with { and end with }. This is critical."

def build_cli_args(messages: List[Message], model: str, stream: bool) -> tuple:
    """Build claude CLI args and stdin prompt."""
    system = next((m.content for m in messages if m.role == "system"), None)

    # Detect if caller expects JSON (system prompt mentions JSON format)
    wants_json = system and ("json" in system.lower() or "JSON" in system)

    conversation = []
    for m in messages:
        if m.role == "system":
            continue
        prefix = "Human" if m.role == "user" else "Assistant"
        conversation.append(f"{prefix}: {m.content}")
    prompt = "\n\n".join(conversation)

    fmt = "stream-json" if stream else "json"
    cmd = ["claude", "-p", "--output-format", fmt, "--model", resolve_model(model),
           "--disable-slash-commands", "--no-session-persistence", '--tools', '']
    if system:
        effective_system = system + (JSON_ENFORCEMENT if wants_json else "")
        cmd.extend(["--system-prompt", effective_system])

    return cmd, prompt

async def call_claude(messages: list[Message], model: str) -> dict:
    """Call claude -p and return parsed JSON result."""
    cmd, prompt = build_cli_args(messages, model, stream=False)
    log.info(f"calling: {' '.join(cmd[:6])}... ({len(prompt)} chars)")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/tmp",  # Avoid loading project CLAUDE.md
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode()),
            timeout=TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(504, f"claude CLI timed out after {TIMEOUT_SECONDS}s")

    if proc.returncode != 0:
        err = stderr.decode().strip()
        log.error(f"claude CLI failed (rc={proc.returncode}): {err}")
        raise HTTPException(502, f"claude CLI error: {err}")

    try:
        result = json.loads(stdout.decode())
    except json.JSONDecodeError:
        raw = stdout.decode()[:500]
        log.error(f"invalid JSON from claude: {raw}")
        raise HTTPException(502, f"invalid JSON from claude CLI")

    if result.get("is_error"):
        raise HTTPException(502, f"claude error: {result.get('result', 'unknown')}")

    return result

async def stream_claude(messages: list[Message], model: str) -> AsyncGenerator[str, None]:
    """Stream claude -p output as SSE chunks in OpenAI format."""
    cmd, prompt = build_cli_args(messages, model, stream=True)
    log.info(f"streaming: {' '.join(cmd[:6])}... ({len(prompt)} chars)")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/tmp",  # Avoid loading project CLAUDE.md
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

        # claude stream-json emits various event types
        if event.get("type") == "assistant" and "content" in event:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": event["content"]},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        elif event.get("type") == "result":
            # Final event — send finish chunk
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"
    await proc.wait()

import re

def _clean_llm_content(text: str) -> str:
    """Clean LLM response: extract JSON from markdown fences if present."""
    # Remove ```json ... ``` wrapping
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if match:
        return match.group(1).strip()
    # If text starts with { or [, it's already clean
    stripped = text.strip()
    if stripped.startswith('{') or stripped.startswith('['):
        return stripped
    # Try to find JSON object in the text
    json_start = text.find('{')
    json_end = text.rfind('}')
    if json_start >= 0 and json_end > json_start:
        return text[json_start:json_end + 1]
    return text

def to_openai_response(result: dict, model: str) -> dict:
    """Convert claude JSON result to OpenAI chat completion format."""
    usage = result.get("usage", {})
    input_tokens = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    content = _clean_llm_content(result.get("result", ""))

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

app = FastAPI(title="Claude CLI Adapter", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "ok", "max_concurrent": MAX_CONCURRENT}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": 0, "owned_by": m["owned_by"]}
            for m in AVAILABLE_MODELS
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    async with semaphore:
        if req.stream:
            return StreamingResponse(
                stream_claude(req.messages, req.model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await call_claude(req.messages, req.model)
        response = to_openai_response(result, req.model)

        cost = result.get("total_cost_usd", 0)
        duration = result.get("duration_ms", 0)
        log.info(f"done: {response['usage']['total_tokens']} tokens, ${cost:.4f}, {duration}ms")

        return response

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    log.info(f"Claude CLI Adapter starting on http://{HOST}:{PORT}")
    log.info(f"Models: {[m['id'] for m in AVAILABLE_MODELS]}")
    log.info(f"Max concurrent: {MAX_CONCURRENT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
