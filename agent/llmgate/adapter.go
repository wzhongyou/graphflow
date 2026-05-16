// Package llmgate adapts github.com/wzhongyou/llmgate to the graphflow agent.LLMModel interface.
package llmgate

import (
	"context"
	"encoding/json"

	"github.com/wzhongyou/llmgate/core"
	"github.com/wzhongyou/llmgate/sdk"

	"github.com/wzhongyou/graphflow/agent"
)

// Adapter implements agent.LLMModel via an llmgate Gateway.
type Adapter struct {
	gw       *sdk.Gateway
	provider string   // provider name for routing; empty = use gateway strategy
	model    string   // optional API model ID override; empty = provider default
	fallback []string // fallback providers (pinned mode only)
}

// Config controls how the Adapter is created.
type Config struct {
	// Provider is the provider name for routing (e.g. "deepseek").
	// When empty, the gateway's strategy config is used.
	Provider string
	// Model is an optional API model ID override (e.g. "deepseek-v4-flash").
	// When empty, the provider's default model is used.
	Model string
	// Fallback providers are tried in order if the primary fails (pinned mode only).
	Fallback []string
}

// New creates an adapter with the given config.
func New(gw *sdk.Gateway, cfg Config) *Adapter {
	return &Adapter{
		gw:       gw,
		provider: cfg.Provider,
		model:    cfg.Model,
		fallback: cfg.Fallback,
	}
}

// NewWithStrategy creates an adapter that routes via the gateway's strategy config.
func NewWithStrategy(gw *sdk.Gateway) *Adapter {
	return &Adapter{gw: gw}
}

// NewAdapter is a convenience that creates a Gateway from a config file.
// cfg.Provider = "" means use strategy.
func NewAdapter(configPath string, cfg Config) (*Adapter, error) {
	gw, err := sdk.NewFromFile(configPath)
	if err != nil {
		return nil, err
	}
	return New(gw, cfg), nil
}

// Chat sends a ChatRequest and returns the response.
func (a *Adapter) Chat(ctx context.Context, req *agent.ChatRequest) (*agent.ChatResponse, error) {
	llmReq := a.buildRequest(req)
	resp, err := a.route(ctx, llmReq)
	if err != nil {
		return nil, err
	}
	return convertResponse(resp), nil
}

// ChatStream sends a streaming ChatRequest and returns a channel of chunks.
func (a *Adapter) ChatStream(ctx context.Context, req *agent.ChatRequest) (<-chan *agent.StreamChunk, error) {
	llmReq := a.buildRequest(req)
	llmReq.Stream = true

	llmCh, err := a.routeStream(ctx, llmReq)
	if err != nil {
		return nil, err
	}

	ch := make(chan *agent.StreamChunk, 8)
	go func() {
		defer close(ch)
		for {
			select {
			case <-ctx.Done():
				return
			case c, ok := <-llmCh:
				if !ok {
					return
				}
				select {
				case ch <- convertChunk(c):
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	return ch, nil
}

func (a *Adapter) buildRequest(req *agent.ChatRequest) *core.ChatRequest {
	return &core.ChatRequest{
		Model:        a.model,
		Messages:     convertMessages(req.Messages),
		Tools:        convertToolDefs(req.Tools),
		Temperature:  req.Temperature,
		MaxTokens:    req.MaxTokens,
		ThinkingType: req.ThinkingType,
	}
}

func (a *Adapter) route(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	if a.provider != "" {
		if len(a.fallback) > 0 {
			all := append([]string{a.provider}, a.fallback...)
			return a.gw.Fallback(all...).Chat(ctx, req)
		}
		return a.gw.With(a.provider).Chat(ctx, req)
	}
	return a.gw.Chat(ctx, req)
}

func (a *Adapter) routeStream(ctx context.Context, req *core.ChatRequest) (<-chan core.StreamChunk, error) {
	if a.provider != "" {
		if len(a.fallback) > 0 {
			all := append([]string{a.provider}, a.fallback...)
			return a.gw.Fallback(all...).ChatStream(ctx, req)
		}
		return a.gw.With(a.provider).ChatStream(ctx, req)
	}
	return a.gw.ChatStream(ctx, req)
}

// ── message conversion ──────────────────────────────────────────────────────────

func convertMessages(msgs []agent.Message) []core.Message {
	out := make([]core.Message, len(msgs))
	for i, m := range msgs {
		out[i] = core.Message{
			Role:             string(m.Role),
			Content:          m.Content,
			ReasoningContent: m.ReasoningContent,
		}
		for _, tc := range m.ToolCalls {
			out[i].ToolCalls = append(out[i].ToolCalls, convertAgentToolCall(tc))
		}
		if m.Role == agent.RoleTool {
			out[i].ToolCallID = m.ToolCallID
		}
	}
	return out
}

func convertAgentToolCall(tc agent.ToolCall) core.ToolCall {
	argsJSON, _ := json.Marshal(tc.Arguments)
	return core.ToolCall{
		ID:   tc.ID,
		Type: "function",
		Function: core.FunctionCall{
			Name:      tc.Name,
			Arguments: string(argsJSON),
		},
	}
}

func convertToolDefs(defs []agent.ToolDef) []core.Tool {
	if len(defs) == 0 {
		return nil
	}
	out := make([]core.Tool, len(defs))
	for i, d := range defs {
		out[i] = core.Tool{
			Type: "function",
			Function: core.ToolFunction{
				Name:        d.Name,
				Description: d.Description,
				Parameters:  d.Parameters,
			},
		}
	}
	return out
}

// ── response conversion ────────────────────────────────────────────────────────

func convertResponse(resp *core.ChatResponse) *agent.ChatResponse {
	if resp == nil {
		return nil
	}
	var toolCalls []agent.ToolCall
	for _, tc := range resp.ToolCalls {
		toolCalls = append(toolCalls, convertCoreToolCall(tc))
	}
	return &agent.ChatResponse{
		Content:          resp.Content,
		ReasoningContent: resp.ReasoningContent,
		ToolCalls:        toolCalls,
		FinishReason:     resp.FinishReason,
		Usage:            convertUsage(&resp.Usage),
	}
}

func convertChunk(c core.StreamChunk) *agent.StreamChunk {
	chunk := &agent.StreamChunk{
		Content:          c.Content,
		ReasoningContent: c.ReasoningContent,
		FinishReason:     c.FinishReason,
		Usage:            convertUsage(c.Usage),
		Error:            c.Error,
	}
	for _, tc := range c.ToolCalls {
		chunk.ToolCalls = append(chunk.ToolCalls, convertCoreToolCall(tc))
	}
	return chunk
}

func convertCoreToolCall(tc core.ToolCall) agent.ToolCall {
	var args map[string]any
	if tc.Function.Arguments != "" {
		json.Unmarshal([]byte(tc.Function.Arguments), &args)
	}
	return agent.ToolCall{
		ID:        tc.ID,
		Name:      tc.Function.Name,
		Arguments: args,
	}
}

func convertUsage(u *core.Usage) *agent.Usage {
	if u == nil {
		return nil
	}
	return &agent.Usage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens,
	}
}
