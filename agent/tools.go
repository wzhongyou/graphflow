package agent

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
)

// Tool is the interface for any capability a ToolNode can invoke.
type Tool interface {
	Name() string
	Description() string
	// Parameters returns a JSON Schema describing the tool's argument object.
	Parameters() map[string]any
	Execute(ctx context.Context, args map[string]any) (string, error)
}

// ToolRegistry holds named tools available to agent nodes.
type ToolRegistry struct {
	tools map[string]Tool
}

// NewToolRegistry creates an empty registry.
func NewToolRegistry() *ToolRegistry { return &ToolRegistry{tools: make(map[string]Tool)} }

// Register adds a tool to the registry.
func (r *ToolRegistry) Register(tool Tool) { r.tools[tool.Name()] = tool }

// Get retrieves a tool by name.
func (r *ToolRegistry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// List returns all registered tools.
func (r *ToolRegistry) List() []Tool {
	out := make([]Tool, 0, len(r.tools))
	for _, t := range r.tools {
		out = append(out, t)
	}
	return out
}

// Len returns the number of registered tools.
func (r *ToolRegistry) Len() int { return len(r.tools) }

// ── Built-in tools ──────────────────────────────────────────────────────────────

// CalculatorTool evaluates arithmetic expressions.
type CalculatorTool struct{}

func (c *CalculatorTool) Name() string        { return "calculator" }
func (c *CalculatorTool) Description() string { return "Evaluate a mathematical expression. Supports +, -, *, /, and parentheses." }
func (c *CalculatorTool) Parameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"expression": map[string]any{
				"type":        "string",
				"description": "Mathematical expression to evaluate, e.g. '(3 + 4) * 5'",
			},
		},
		"required": []string{"expression"},
	}
}

func (c *CalculatorTool) Execute(_ context.Context, args map[string]any) (string, error) {
	expr, ok := args["expression"]
	if !ok {
		return "", fmt.Errorf("calculator: missing 'expression' argument")
	}
	exprStr, ok := expr.(string)
	if !ok {
		return "", fmt.Errorf("calculator: 'expression' must be a string")
	}
	result, err := evalExpr(exprStr)
	if err != nil {
		return "", fmt.Errorf("calculator: %w", err)
	}
	return strconv.FormatFloat(result, 'f', -1, 64), nil
}

// evalExpr parses and evaluates a simple arithmetic expression.
func evalExpr(expr string) (float64, error) {
	tree, err := parser.ParseExpr(expr)
	if err != nil {
		return 0, fmt.Errorf("parse %q: %w", expr, err)
	}
	return evalNode(tree)
}

func evalNode(n ast.Expr) (float64, error) {
	switch e := n.(type) {
	case *ast.BasicLit:
		if e.Kind == token.INT || e.Kind == token.FLOAT {
			return strconv.ParseFloat(e.Value, 64)
		}
		return 0, fmt.Errorf("unsupported literal: %s", e.Value)
	case *ast.BinaryExpr:
		left, err := evalNode(e.X)
		if err != nil {
			return 0, err
		}
		right, err := evalNode(e.Y)
		if err != nil {
			return 0, err
		}
		switch e.Op {
		case token.ADD:
			return left + right, nil
		case token.SUB:
			return left - right, nil
		case token.MUL:
			return left * right, nil
		case token.QUO:
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			return left / right, nil
		default:
			return 0, fmt.Errorf("unsupported operator: %s", e.Op)
		}
	case *ast.ParenExpr:
		return evalNode(e.X)
	case *ast.UnaryExpr:
		val, err := evalNode(e.X)
		if err != nil {
			return 0, err
		}
		if e.Op == token.SUB {
			return -val, nil
		}
		return val, nil
	default:
		return 0, fmt.Errorf("unsupported expression type: %T", n)
	}
}
