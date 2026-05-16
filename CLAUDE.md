# CLAUDE.md

## Project Overview

Graphflow is a Go-native Agent development framework. The graph execution engine (`graph/`) is the AI-agnostic core; the Agent layer (`agent/`) builds on top of it. LLM access goes through the [llmgate](https://github.com/wzhongyou/llmgate) gateway (`agent/llmgate/` adapter).

## Essential Commands

```bash
go build ./...          # build all packages
go vet ./...            # static analysis (must pass before any commit)
go test ./...           # run all tests
go test ./agent/...     # run agent tests only
go test ./graph/...     # run graph engine tests only

# Run agent demo (mock mode, no API key needed)
go run ./examples/agent_demo

# Run agent demo with real LLM (requires config/llmgate.toml)
go run ./examples/agent_demo -q "Calculate 100 + 200"
```

## Package Structure

```
graph/          # Core engine — Graph[S], Engine[S], NodeFunc[S]
  middleware/   # NodeFunc decorators (retry, timeout, circuit breaker, …)
  node/         # Built-in nodes (HTTP, Delay, Transform, Noop)
  checkpoint/   # Persistence backends (memory, file, redis, sqlite)
agent/          # Agent abstractions — MessageState, LLMNode, ToolNode, …
  llmgate/      # llmgate SDK adapter (implements agent.LLMModel)
config/         # Configuration templates (llmgate.toml.example)
examples/
  agent_demo/   # ReAct agent with Hook tracing (mock + real LLM)
docs/           # Design docs and guides
```

## Key Conventions

- `graph/` has **zero external dependencies** — do not add any imports outside stdlib.
- `agent/` may import `graph/` but never vice versa.
- `agent/llmgate/` imports `github.com/wzhongyou/llmgate` (the only external dep for agent layer).
- External integrations (Redis, SQLite, OTel) go in sub-packages so the core stays dependency-free.
- All public types use Go generics (`Graph[S]`, `Engine[S]`, `Hook[S]`) — keep the state type `S` as the single generic parameter per graph.
- Node functions match exactly: `func(ctx context.Context, state S) (S, error)`.
- Middleware wraps `NodeFunc[S]` and returns `NodeFunc[S]` — composable by design.

## Implementation Status

Use `TODO(Px)` / `TODO(Ax)` markers that match the roadmap phases in the design doc (`docs/graphflow-design.md`):

| Marker | Meaning | Status |
|--------|---------|--------|
| `TODO(P1)` | Core P1 — graph model, sequential engine, YAML config | Core done; YAML pending |
| `TODO(P7)` | OTel hook | Not started |
| `TODO(A8)` | SupervisorAgent.BuildGraph | Stub only |
| `TODO(A9)+` | Streaming, Structured Output, etc. | Not started |

Phases **A1–A7 are complete** (MessageState, LLMModel, LLMNode, ToolNode, ToolRegistry, CalculatorTool, ShortTermMemory, LongTermMemory, ReActAgent, RAGAgent, llmgate adapter).

## LLM Configuration

llmgate config goes in `config/llmgate.toml` (gitignored). Template at `config/llmgate.toml.example`. Three setup modes:

1. Config file: auto-detected from `config/llmgate.toml` or `llmgate.toml`
2. Env vars: `DEEPSEEK_KEY=sk-xxx` (auto-detected)
3. Mock: falls back when no config or keys found

## Design Decisions

See `docs/graphflow-design.md` for full rationale. Key ones:

- **Pregel-style execution**: superstep loop, not recursive calls.
- **Back edges = loops**: detected by DFS at `Compile()` time; `SetMaxIterations` guards against infinite loops (default 1000).
- **Conditional edges take priority** over unconditional edges; first match wins.
- **Multiple unconditional edges = fan-out** (parallel, implemented in `engine_parallel.go`).
- **Hook is stored as `any`** in `runConfig` and type-asserted in `hookOf[S]` — avoids making `Option` generic at the cost of a silent no-op on type mismatch.
- **LLMModel interface is provider-agnostic**; the llmgate adapter handles provider differences (OpenAI/Anthropic/Gemini protocol mapping, fallback, strategy routing).
