package llmgate

import (
	"encoding/json"
	"testing"

	"github.com/wzhongyou/llmgate/core"

	"github.com/wzhongyou/graphflow/agent"
)

func TestConvertMessages_Plain(t *testing.T) {
	msgs := []agent.Message{
		{Role: agent.RoleUser, Content: "hello"},
		{Role: agent.RoleAssistant, Content: "hi"},
	}
	out := convertMessages(msgs)
	if len(out) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out))
	}
	if out[0].Role != "user" || out[0].Content != "hello" {
		t.Fatalf("unexpected message[0]: %+v", out[0])
	}
}

func TestConvertMessages_ToolCalls(t *testing.T) {
	msgs := []agent.Message{
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
			{ID: "call_1", Name: "calc", Arguments: map[string]any{"expr": "1+1"}},
		}},
	}
	out := convertMessages(msgs)
	if len(out[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(out[0].ToolCalls))
	}
	if out[0].ToolCalls[0].Function.Name != "calc" {
		t.Fatalf("unexpected tool call name: %s", out[0].ToolCalls[0].Function.Name)
	}
	var args map[string]any
	json.Unmarshal([]byte(out[0].ToolCalls[0].Function.Arguments), &args)
	if args["expr"] != "1+1" {
		t.Fatalf("unexpected args: %v", args)
	}
}

func TestConvertMessages_ToolResult(t *testing.T) {
	msgs := []agent.Message{
		{Role: agent.RoleTool, ToolCallID: "call_1", ToolName: "calc", Content: "result: 2"},
	}
	out := convertMessages(msgs)
	if out[0].Role != "tool" {
		t.Fatalf("expected role=tool, got %s", out[0].Role)
	}
	if out[0].ToolCallID != "call_1" {
		t.Fatalf("expected ToolCallID=call_1, got %s", out[0].ToolCallID)
	}
}

func TestConvertToolDefs_Nil(t *testing.T) {
	out := convertToolDefs(nil)
	if out != nil {
		t.Fatalf("expected nil for nil input, got %v", out)
	}
}

func TestConvertToolDefs_Empty(t *testing.T) {
	out := convertToolDefs([]agent.ToolDef{})
	if out != nil {
		t.Fatalf("expected nil for empty input, got %v", out)
	}
}

func TestConvertToolDefs_WithParams(t *testing.T) {
	defs := []agent.ToolDef{
		{Name: "search", Description: "search the web", Parameters: map[string]any{"type": "object"}},
	}
	out := convertToolDefs(defs)
	if len(out) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(out))
	}
	if out[0].Type != "function" {
		t.Fatalf("expected type=function, got %s", out[0].Type)
	}
	if out[0].Function.Name != "search" {
		t.Fatalf("expected name=search, got %s", out[0].Function.Name)
	}
}

func TestConvertResponse_Plain(t *testing.T) {
	resp := &core.ChatResponse{
		Content:      "hello",
		FinishReason: "stop",
		Model:        "test",
		Usage:        core.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}
	out := convertResponse(resp)
	if out.Content != "hello" {
		t.Fatalf("expected 'hello', got %q", out.Content)
	}
	if out.FinishReason != "stop" {
		t.Fatalf("expected stop, got %s", out.FinishReason)
	}
	if out.Usage == nil {
		t.Fatal("expected non-nil usage")
	}
	if out.Usage.TotalTokens != 15 {
		t.Fatalf("expected total=15, got %d", out.Usage.TotalTokens)
	}
}

func TestConvertResponse_ToolCalls(t *testing.T) {
	resp := &core.ChatResponse{
		FinishReason: "tool_calls",
		ToolCalls: []core.ToolCall{
			{ID: "tc1", Type: "function", Function: core.FunctionCall{Name: "calc", Arguments: `{"expr":"1+1"}`}},
		},
	}
	out := convertResponse(resp)
	if len(out.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(out.ToolCalls))
	}
	if out.ToolCalls[0].Name != "calc" {
		t.Fatalf("expected calc, got %s", out.ToolCalls[0].Name)
	}
	if out.ToolCalls[0].Arguments["expr"] != "1+1" {
		t.Fatalf("unexpected args: %v", out.ToolCalls[0].Arguments)
	}
}

func TestConvertResponse_Nil(t *testing.T) {
	out := convertResponse(nil)
	if out != nil {
		t.Fatal("expected nil for nil input")
	}
}

func TestConvertUsage_Nil(t *testing.T) {
	out := convertUsage(nil)
	if out != nil {
		t.Fatal("expected nil for nil input")
	}
}

func TestConvertChunk_WithToolCalls(t *testing.T) {
	c := core.StreamChunk{
		Content:      "",
		FinishReason: "tool_calls",
		ToolCalls: []core.ToolCall{
			{ID: "tc1", Type: "function", Function: core.FunctionCall{Name: "search", Arguments: `{}`}},
		},
		Usage: &core.Usage{TotalTokens: 100},
	}
	out := convertChunk(c)
	if out.Content != "" {
		t.Fatalf("expected empty content, got %q", out.Content)
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(out.ToolCalls))
	}
	if out.Usage == nil || out.Usage.TotalTokens != 100 {
		t.Fatalf("unexpected usage: %+v", out.Usage)
	}
}

func TestConvertAgentToolCall_RoundTrip(t *testing.T) {
	orig := agent.ToolCall{
		ID:   "call_42",
		Name: "test_tool",
		Arguments: map[string]any{
			"key":  "value",
			"num":  float64(42),
			"bool": true,
		},
	}
	coreTC := convertAgentToolCall(orig)
	back := convertCoreToolCall(coreTC)
	if back.ID != orig.ID {
		t.Fatalf("ID mismatch: %s != %s", back.ID, orig.ID)
	}
	if back.Name != orig.Name {
		t.Fatalf("Name mismatch: %s != %s", back.Name, orig.Name)
	}
	if back.Arguments["key"] != "value" {
		t.Fatalf("key mismatch: %v", back.Arguments["key"])
	}
	if back.Arguments["num"] != float64(42) {
		t.Fatalf("num mismatch: %v", back.Arguments["num"])
	}
}

func TestAdapter_buildRequest(t *testing.T) {
	a := New(nil, Config{Provider: "test-provider", Model: "test-model"})
	temp := 0.7
	maxTok := 2048
	req := &agent.ChatRequest{
		Messages:     []agent.Message{{Role: agent.RoleUser, Content: "hi"}},
		Temperature:  &temp,
		MaxTokens:    &maxTok,
		ThinkingType: "disabled",
	}
	llmReq := a.buildRequest(req)
	if llmReq.Model != "test-model" {
		t.Fatalf("expected model=test-model, got %s", llmReq.Model)
	}
	if llmReq.Temperature == nil || *llmReq.Temperature != 0.7 {
		t.Fatalf("expected temperature=0.7, got %v", llmReq.Temperature)
	}
	if llmReq.MaxTokens == nil || *llmReq.MaxTokens != 2048 {
		t.Fatalf("expected maxTokens=2048, got %v", llmReq.MaxTokens)
	}
	if len(llmReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(llmReq.Messages))
	}
	if llmReq.ThinkingType != "disabled" {
		t.Fatalf("expected ThinkingType=disabled, got %s", llmReq.ThinkingType)
	}
}

func TestConvertMessages_ReasoningContent(t *testing.T) {
	msgs := []agent.Message{
		{Role: agent.RoleAssistant, ReasoningContent: "Thinking step by step...", Content: "answer"},
	}
	out := convertMessages(msgs)
	if out[0].ReasoningContent != "Thinking step by step..." {
		t.Fatalf("expected reasoning content, got %q", out[0].ReasoningContent)
	}
}

func TestConvertResponse_ReasoningContent(t *testing.T) {
	resp := &core.ChatResponse{
		ReasoningContent: "reasoning...",
		Content:          "final answer",
	}
	out := convertResponse(resp)
	if out.ReasoningContent != "reasoning..." {
		t.Fatalf("expected reasoning='reasoning...', got %q", out.ReasoningContent)
	}
}
