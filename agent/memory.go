package agent

import (
	"context"
	"fmt"
)

// ShortTermMemory keeps the recent conversation window in memory.
type ShortTermMemory struct {
	maxMessages int
	messages    []Message
}

// NewShortTermMemory creates a memory buffer capped at maxMessages.
func NewShortTermMemory(maxMessages int) *ShortTermMemory {
	return &ShortTermMemory{maxMessages: maxMessages}
}

// Add appends a message, evicting the oldest if the buffer exceeds maxMessages.
func (m *ShortTermMemory) Add(msg Message) {
	m.messages = append(m.messages, msg)
	if len(m.messages) > m.maxMessages {
		// Evict oldest non-system messages first
		idx := 0
		for idx < len(m.messages) && len(m.messages) > m.maxMessages {
			if m.messages[idx].Role != RoleSystem {
				m.messages = append(m.messages[:idx], m.messages[idx+1:]...)
			} else {
				idx++
			}
		}
		// If still over limit after preserving system messages, trim from front
		if len(m.messages) > m.maxMessages {
			excess := len(m.messages) - m.maxMessages
			m.messages = m.messages[excess:]
		}
	}
}

// Messages returns the current message window.
func (m *ShortTermMemory) Messages() []Message { return m.messages }

// LongTermMemory persists and retrieves memories via a VectorStore.
type LongTermMemory struct {
	embedder    Embedder
	vectorStore VectorStore
}

// NewLongTermMemory creates a long-term memory backed by the given stores.
func NewLongTermMemory(embedder Embedder, store VectorStore) *LongTermMemory {
	return &LongTermMemory{embedder: embedder, vectorStore: store}
}

// Remember embeds and stores a memory string.
func (m *LongTermMemory) Remember(ctx context.Context, text string, metadata map[string]any) error {
	if m.embedder == nil || m.vectorStore == nil {
		return fmt.Errorf("long-term memory: embedder and vectorStore must be set")
	}
	vector, err := m.embedder.Embed(ctx, text)
	if err != nil {
		return fmt.Errorf("embedding: %w", err)
	}
	id := fmt.Sprintf("mem-%d", len(vector)) // placeholder ID; caller should supply
	_ = id
	return m.vectorStore.Insert(ctx, id, vector, metadata)
}

// Recall retrieves the top-k most relevant memories for a query.
func (m *LongTermMemory) Recall(ctx context.Context, query string, topK int) ([]SearchResult, error) {
	if m.embedder == nil || m.vectorStore == nil {
		return nil, fmt.Errorf("long-term memory: embedder and vectorStore must be set")
	}
	vector, err := m.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding: %w", err)
	}
	return m.vectorStore.Search(ctx, vector, topK)
}
