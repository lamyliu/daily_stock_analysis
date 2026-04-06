# Claude CLI → OpenAI 兼容 API 适配器

## 目标

将 Claude Code 的 `claude -p` 管道模式包装为 OpenAI 兼容的 HTTP API，让任何支持 OpenAI SDK 的项目（daily_stock_analysis、LangChain、Cursor 等）直接使用 Claude 订阅，无需 API Key。

**本质：把 Claude 订阅变成本地 OpenAI 兼容 API。**

## 架构

```
任意 OpenAI SDK 客户端
    ↓ POST /v1/chat/completions（标准 OpenAI 格式）
Claude CLI Adapter（127.0.0.1:8877）
    ↓ asyncio.create_subprocess: claude -p --system-prompt "..." --model xxx --output-format json
Claude Code CLI（自动管理订阅认证）
    ↓
Anthropic API
    ↓
JSON 响应（含 result + usage + cost）→ 转换为 OpenAI 格式 → 返回
```

## claude -p 能力探测

实测 `claude -p --output-format json` 返回的数据非常丰富：

```json
{
  "type": "result",
  "result": "回复内容",
  "duration_ms": 2204,
  "stop_reason": "end_turn",
  "total_cost_usd": 0.038,
  "usage": {
    "input_tokens": 2,
    "output_tokens": 4,
    "cache_creation_input_tokens": 10235,
    "cache_read_input_tokens": 0
  }
}
```

关键发现：
- `--system-prompt` — 原生支持 system message，不需要拼接
- `--output-format json` — 返回结构化 JSON，含完整 token 用量和成本
- `--output-format stream-json` — 支持流式输出
- `--model sonnet/opus/haiku` — 支持别名和完整模型名
- `--bare` — 跳过 hooks/CLAUDE.md/内存等，纯净 API 调用（更快）
- usage 包含 `cache_creation_input_tokens` 和 `cache_read_input_tokens`（prompt cache 信息）

## API 端点

```
POST /v1/chat/completions    # 核心：聊天补全
GET  /v1/models              # 模型列表
GET  /health                 # 健康检查
```

## 请求/响应格式

### 请求（OpenAI 标准）

```json
{
  "model": "claude-sonnet-4-5",
  "messages": [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"}
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "stream": false
}
```

### 响应（OpenAI 标准）

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "claude-sonnet-4-5",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "你好！"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10237,
    "completion_tokens": 4,
    "total_tokens": 10241
  }
}
```

> usage 数据来自 claude CLI 的 JSON 输出，不是估算。

## 核心实现

### 消息处理

```python
def build_cli_args(messages: list, model: str, stream: bool) -> tuple[list[str], str]:
    """构建 claude CLI 参数和 stdin 输入"""
    # 提取 system message
    system = next((m["content"] for m in messages if m["role"] == "system"), None)

    # 拼接 user/assistant 轮次为对话（保留多轮上下文）
    conversation = []
    for m in messages:
        if m["role"] == "system":
            continue
        prefix = "Human" if m["role"] == "user" else "Assistant"
        conversation.append(f"{prefix}: {m['content']}")
    prompt = "\n\n".join(conversation)

    # 构建命令行
    fmt = "stream-json" if stream else "json"
    cmd = ["claude", "-p", "--bare", "--output-format", fmt, "--model", model]
    if system:
        cmd.extend(["--system-prompt", system])

    return cmd, prompt
```

### 异步子进程（非阻塞）

```python
async def call_claude(messages, model, stream=False):
    cmd, prompt = build_cli_args(messages, model, stream)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(prompt.encode()),
        timeout=300,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI error: {stderr.decode()}")

    result = json.loads(stdout.decode())
    return result
```

### 并发控制

```python
# Claude 订阅有速率限制，用信号量控制最大并发
MAX_CONCURRENT = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

async def chat_completion(request):
    async with semaphore:
        result = await call_claude(request.messages, request.model, request.stream)
        return to_openai_response(result)
```

### Streaming

```python
async def stream_claude(messages, model):
    """SSE 流式输出，兼容 OpenAI streaming 格式"""
    cmd, prompt = build_cli_args(messages, model, stream=True)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    proc.stdin.write(prompt.encode())
    proc.stdin.close()

    async for line in proc.stdout:
        text = line.decode().strip()
        if not text:
            continue
        try:
            event = json.loads(text)
            if event.get("type") == "assistant" and "content" in event:
                chunk = to_openai_stream_chunk(event["content"])
                yield f"data: {json.dumps(chunk)}\n\n"
        except json.JSONDecodeError:
            continue

    yield "data: [DONE]\n\n"
```

## 与 daily_stock_analysis 集成

### 方式一：自定义渠道（推荐）

```env
LLM_CHANNELS=claude_cli
LLM_CLAUDE_CLI_BASE_URL=http://127.0.0.1:8877/v1
LLM_CLAUDE_CLI_API_KEY=dummy
LLM_CLAUDE_CLI_MODELS=claude-sonnet-4-5,claude-opus-4-6,claude-haiku-4-5
LLM_CLAUDE_CLI_PROTOCOL=openai
```

### 方式二：OPENAI 兼容（最简单）

```env
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://127.0.0.1:8877/v1
LITELLM_MODEL=openai/claude-sonnet-4-5
```

### Docker 场景

容器内无法直接调用宿主机的 `claude` CLI。解决方案：

```env
# 适配器在宿主机运行，容器通过 host.docker.internal 访问
LLM_CLAUDE_CLI_BASE_URL=http://host.docker.internal:8877/v1
```

## 文件结构

```
scripts/
└── claude-cli-adapter.py    # 单文件，约 200 行
```

## 启动

```bash
# 前台运行
python scripts/claude-cli-adapter.py

# 后台运行
nohup python scripts/claude-cli-adapter.py > /tmp/claude-cli-adapter.log 2>&1 &

# 或 LaunchAgent 开机自启
```

### LaunchAgent

```xml
<!-- ~/Library/LaunchAgents/ai.claude-cli-adapter.plist -->
<plist version="1.0">
<dict>
    <key>Label</key><string>ai.claude-cli-adapter</string>
    <key>ProgramArguments</key>
    <array>
        <string>python3</string>
        <string>/Users/darkmagician/.openclaw/workspace/daily_stock_analysis/scripts/claude-cli-adapter.py</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/claude-cli-adapter.log</string>
    <key>StandardErrorPath</key><string>/tmp/claude-cli-adapter.err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/Users/darkmagician/.nvm/versions/node/v22.22.0/bin</string>
    </dict>
</dict>
</plist>
```

## 通用价值

这个适配器不只服务 daily_stock_analysis，**任何 `OPENAI_BASE_URL` 可配置的项目都能用**：

| 项目 | 配置方式 |
|------|---------|
| daily_stock_analysis | `LLM_CHANNELS=claude_cli` |
| LangChain | `ChatOpenAI(base_url="http://127.0.0.1:8877/v1")` |
| Cursor | Settings → Models → OpenAI API Base |
| LlamaIndex | `OpenAI(api_base="http://127.0.0.1:8877/v1")` |
| 知识库 ingest.ts | 替换 callLLM() 用 fetch 调本地 API |
| Continue (IDE) | config.json → apiBase |

## 限制

| 限制 | 说明 | 缓解 |
|------|------|------|
| 并发 | Claude 订阅有速率限制 | Semaphore(3) 控制并发 |
| 延迟 | 子进程启动 ~200ms 开销 | `--bare` 跳过 hooks 加速 |
| 仅限本机 | 监听 127.0.0.1 | 安全设计，不暴露网络 |
| 无多轮状态 | 每次调用独立 | messages 里包含历史即可 |
| 5h 窗口 | Claude 订阅滚动限额 | 高峰时段降级到 Haiku |

## 与备选方案的对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Claude CLI 适配器（本方案）** | 免费用订阅、token 用量精确、支持 streaming | 仅限本机、有并发限制 |
| Anthropic API Key | 无并发限制、可多机 | 额外付费 |
| MiniMax API | 国内低延迟、便宜 | 质量不如 Claude |
| Ollama 本地 | 完全免费、无限制 | 需要 GPU、质量有限 |

**推荐组合**：Claude CLI 适配器为主 + MiniMax 为 fallback（订阅额度用完时自动切换）。
