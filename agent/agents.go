package agent

import (
	"context"
	"fmt"

	"github.com/wzhongyou/graphflow/graph"
)

// ── ReActAgent ──────────────────────────────────────────────────────────────────

// ReActAgentConfig configures a ReAct-style agent.
type ReActAgentConfig struct {
	Name         string
	LLM          LLMModel
	SystemPrompt string
	Tools        []Tool
	MaxSteps     int
}

// ReActAgent builds a Reason-Act loop graph.
type ReActAgent struct{ cfg ReActAgentConfig }

// NewReActAgent creates a ReActAgent.
func NewReActAgent(cfg ReActAgentConfig) *ReActAgent {
	if cfg.Name == "" {
		cfg.Name = "react-agent"
	}
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = 20
	}
	return &ReActAgent{cfg: cfg}
}

// BuildGraph constructs and compiles the ReAct graph.
// Structure: llm ──(has tool calls)──→ tool ──→ llm (loop)
func (a *ReActAgent) BuildGraph() (*graph.Graph[*MessageState], error) {
	llmNode := NewLLMNode(LLMNodeConfig{
		Model:        a.cfg.LLM,
		SystemPrompt: a.cfg.SystemPrompt,
		Tools:        a.cfg.Tools,
	})
	toolNode := NewToolNode(a.cfg.Tools...)

	g := graph.NewGraph[*MessageState](a.cfg.Name)
	g.AddNode("llm", llmNode.Run)
	g.AddNode("tool", toolNode.Run)
	g.SetEntryPoint("llm")

	g.AddCondition("llm", graph.Condition[*MessageState]{
		If:     HasPendingToolCalls,
		Target: "tool",
	})

	g.AddEdge("tool", "llm")
	g.SetMaxIterations("llm", a.cfg.MaxSteps)

	if err := g.Compile(); err != nil {
		return nil, fmt.Errorf("react agent: %w", err)
	}
	return g, nil
}

// HasPendingToolCalls returns true when the last assistant message contains
// tool calls that need executing.
func HasPendingToolCalls(_ context.Context, s *MessageState) bool {
	if len(s.Messages) == 0 {
		return false
	}
	last := s.Messages[len(s.Messages)-1]
	return last.Role == RoleAssistant && len(last.ToolCalls) > 0
}

// ── RAGAgent ────────────────────────────────────────────────────────────────────

// RAGAgentConfig configures a Retrieval-Augmented Generation agent.
type RAGAgentConfig struct {
	Name         string
	LLM          LLMModel
	Embedder     Embedder
	VectorStore  VectorStore
	SystemPrompt string
	TopK         int
}

// RAGAgent builds a retrieve-then-generate graph.
type RAGAgent struct{ cfg RAGAgentConfig }

// NewRAGAgent creates a RAGAgent.
func NewRAGAgent(cfg RAGAgentConfig) *RAGAgent {
	if cfg.Name == "" {
		cfg.Name = "rag-agent"
	}
	if cfg.TopK <= 0 {
		cfg.TopK = 5
	}
	return &RAGAgent{cfg: cfg}
}

// BuildGraph constructs and compiles the RAG graph.
// Structure: retrieve → llm
func (a *RAGAgent) BuildGraph() (*graph.Graph[*MessageState], error) {
	retrieveNode := &VectorRetrieveNode{
		Embedder:    a.cfg.Embedder,
		VectorStore: a.cfg.VectorStore,
		TopK:        a.cfg.TopK,
	}
	llmNode := NewLLMNode(LLMNodeConfig{
		Model:        a.cfg.LLM,
		SystemPrompt: a.cfg.SystemPrompt,
	})

	g := graph.NewGraph[*MessageState](a.cfg.Name)
	g.AddNode("retrieve", retrieveNode.Run)
	g.AddNode("llm", llmNode.Run)
	g.SetEntryPoint("retrieve")
	g.AddEdge("retrieve", "llm")

	if err := g.Compile(); err != nil {
		return nil, fmt.Errorf("rag agent: %w", err)
	}
	return g, nil
}

// ── SupervisorAgent ─────────────────────────────────────────────────────────────

// SupervisorAgentConfig configures a multi-agent supervisor.
type SupervisorAgentConfig struct {
	Name      string
	LLM       LLMModel
	SubAgents map[string]SubAgent
	MaxRounds int
}

// SubAgent is implemented by any agent that can be orchestrated by a supervisor.
type SubAgent interface {
	BuildGraph() (*graph.Graph[*MessageState], error)
	Name() string
}

// SupervisorAgent routes tasks to sub-agents and aggregates results.
type SupervisorAgent struct{ cfg SupervisorAgentConfig }

// NewSupervisorAgent creates a SupervisorAgent.
func NewSupervisorAgent(cfg SupervisorAgentConfig) *SupervisorAgent {
	if cfg.Name == "" {
		cfg.Name = "supervisor-agent"
	}
	if cfg.MaxRounds <= 0 {
		cfg.MaxRounds = 10
	}
	return &SupervisorAgent{cfg: cfg}
}

// BuildGraph constructs the supervisor orchestration graph.
// TODO(A8): supervisor_llm → route → sub-agent subgraphs → aggregate
func (a *SupervisorAgent) BuildGraph() (*graph.Graph[*MessageState], error) {
	return nil, fmt.Errorf("SupervisorAgent: not yet implemented (A8)")
}
