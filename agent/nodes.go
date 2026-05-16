package agent

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// LLMNodeConfig configures an LLMNode.
type LLMNodeConfig struct {
	Model        LLMModel
	SystemPrompt string
	Tools        []Tool
	Temperature  *float64
	MaxTokens    *int
	ThinkingType string // "disabled" to disable reasoning/thinking mode
	Stream       bool
}

// LLMNode calls an LLM and appends the assistant reply to MessageState.Messages.
type LLMNode struct{ cfg LLMNodeConfig }

// NewLLMNode creates an LLMNode with the given config.
func NewLLMNode(cfg LLMNodeConfig) *LLMNode { return &LLMNode{cfg: cfg} }

// Run implements graph.NodeFunc[*MessageState].
func (n *LLMNode) Run(ctx context.Context, s *MessageState) (*MessageState, error) {
	messages := n.buildMessages(s)
	req := &ChatRequest{
		Messages:     messages,
		Tools:        ToolDefs(n.cfg.Tools),
		Temperature:  n.cfg.Temperature,
		MaxTokens:    n.cfg.MaxTokens,
		ThinkingType: n.cfg.ThinkingType,
	}

	var resp *ChatResponse
	var err error
	if n.cfg.Stream {
		resp, err = n.chatStream(ctx, req)
	} else {
		resp, err = n.cfg.Model.Chat(ctx, req)
	}
	if err != nil {
		return s, err
	}

	msg := Message{
		Role:             RoleAssistant,
		Content:          resp.Content,
		ReasoningContent: resp.ReasoningContent,
		ToolCalls:        resp.ToolCalls,
		Timestamp:        time.Now(),
	}
	s.Messages = append(s.Messages, msg)
	s.StepCount++
	if resp.Usage != nil {
		s.TotalTokens += resp.Usage.TotalTokens
	}
	return s, nil
}

func (n *LLMNode) buildMessages(s *MessageState) []Message {
	messages := s.Messages
	if n.cfg.SystemPrompt == "" {
		return messages
	}
	for _, m := range messages {
		if m.Role == RoleSystem {
			return messages
		}
	}
	return append([]Message{{Role: RoleSystem, Content: n.cfg.SystemPrompt}}, messages...)
}

func (n *LLMNode) chatStream(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	chunks, err := n.cfg.Model.ChatStream(ctx, req)
	if err != nil {
		return nil, err
	}
	var content strings.Builder
	var toolCalls []ToolCall
	var usage *Usage
	for chunk := range chunks {
		if chunk.Error != nil {
			return nil, chunk.Error
		}
		content.WriteString(chunk.Content)
		if chunk.ToolCalls != nil {
			toolCalls = chunk.ToolCalls
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	return &ChatResponse{
		Content:   content.String(),
		ToolCalls: toolCalls,
		Usage:     usage,
	}, nil
}

// ToolNodeConfig configures the tool execution behaviour.
type ToolNodeConfig struct {
	Tools    []Tool
	Parallel bool // if true execute tools concurrently
}

// ToolNode executes pending tool calls found in the last assistant message.
type ToolNode struct {
	registry *ToolRegistry
	parallel bool
}

// NewToolNode creates a ToolNode backed by the given tools.
func NewToolNode(tools ...Tool) *ToolNode {
	r := NewToolRegistry()
	for _, t := range tools {
		r.Register(t)
	}
	return &ToolNode{registry: r}
}

// NewToolNodeWithRegistry creates a ToolNode from an existing registry.
func NewToolNodeWithRegistry(registry *ToolRegistry) *ToolNode {
	return &ToolNode{registry: registry}
}

// Run implements graph.NodeFunc[*MessageState].
func (n *ToolNode) Run(ctx context.Context, s *MessageState) (*MessageState, error) {
	if len(s.Messages) == 0 {
		return s, nil
	}
	last := s.Messages[len(s.Messages)-1]
	if last.Role != RoleAssistant || len(last.ToolCalls) == 0 {
		return s, nil
	}

	for _, tc := range last.ToolCalls {
		result := n.executeToolCall(ctx, tc)
		s.Messages = append(s.Messages, Message{
			Role:       RoleTool,
			Content:    result,
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Timestamp:  time.Now(),
		})
	}
	return s, nil
}

func (n *ToolNode) executeToolCall(ctx context.Context, tc ToolCall) string {
	tool, ok := n.registry.Get(tc.Name)
	if !ok {
		return fmt.Sprintf("error: tool %q not found", tc.Name)
	}
	result, err := tool.Execute(ctx, tc.Arguments)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	return result
}

// VectorRetrieveNode embeds the last user message and retrieves relevant documents.
type VectorRetrieveNode struct {
	Embedder    Embedder
	VectorStore VectorStore
	TopK        int
}

// Run implements graph.NodeFunc[*MessageState].
func (n *VectorRetrieveNode) Run(ctx context.Context, s *MessageState) (*MessageState, error) {
	if len(s.Messages) == 0 || n.Embedder == nil || n.VectorStore == nil {
		return s, nil
	}

	last := s.Messages[len(s.Messages)-1]
	vector, err := n.Embedder.Embed(ctx, last.Content)
	if err != nil {
		return s, fmt.Errorf("embedding: %w", err)
	}

	results, err := n.VectorStore.Search(ctx, vector, n.TopK)
	if err != nil {
		return s, fmt.Errorf("vector search: %w", err)
	}

	if s.Context == nil {
		s.Context = make(map[string]any)
	}
	docs := make([]string, len(results))
	for i, r := range results {
		docs[i] = r.ID
	}
	s.Context["retrieved_docs"] = results
	s.Context["retrieved_doc_ids"] = docs
	return s, nil
}

// HumanInputNode suspends the graph and waits for a human message (HITL).
type HumanInputNode struct {
	Prompt  string
	Timeout time.Duration
}

// Run implements graph.NodeFunc[*MessageState].
func (n *HumanInputNode) Run(ctx context.Context, s *MessageState) (*MessageState, error) {
	// HITL requires an external channel/checkpoint mechanism; stub for now.
	return s, nil
}
