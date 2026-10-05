// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package runner_test

import (
	"context"
	"errors"
	"iter"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// countingAsker pauses on RequestInput and, when re-entered with the
// reply, emits it as output. failFirstResume makes its first re-entry
// return an error.
type countingAsker struct {
	workflow.BaseNode
	id              string
	runs            atomic.Int32
	failFirstResume bool
	failed          atomic.Bool
}

func newCountingAsker(name, id string) *countingAsker {
	yes := true
	return &countingAsker{BaseNode: workflow.NewBaseNode(name, "", workflow.NodeConfig{RerunOnResume: &yes}), id: id}
}

func (n *countingAsker) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.runs.Add(1)
		if resp, ok := ctx.ResumedInput(n.id); ok {
			if n.failFirstResume && !n.failed.Swap(true) {
				yield(nil, errTransient)
				return
			}
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = resp
			yield(ev, nil)
			return
		}
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{InterruptID: n.id, Message: n.id}), nil)
	}
}

type transientErr struct{}

func (transientErr) Error() string { return "transient failure" }

var errTransient = transientErr{}

func runReturningErr(t *testing.T, r *runner.Runner, msg *genai.Content) ([]*session.Event, error) {
	t.Helper()
	var evs []*session.Event
	var first error
	for ev, err := range r.Run(t.Context(), nodeTestUser, nodeTestSession, msg, agent.RunConfig{}) {
		if err != nil {
			if first == nil {
				first = err
			}
			continue
		}
		evs = append(evs, ev)
	}
	return evs, first
}

// Two approval steps in sequence. Answering the second must not re-run the first.
func TestResumeRegression_ChainedReentryAskers(t *testing.T) {
	a, b := newCountingAsker("approveA", "a"), newCountingAsker("approveB", "b")
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, a, b))

	t1, _ := runReturningErr(t, r, userText("start"))
	idA, nameA := findLongRunningInterrupt(t1)
	t2, _ := runReturningErr(t, r, resumeContent(idA, nameA, "ok-a"))
	idB, nameB := findLongRunningInterrupt(t2)
	if _, err := runReturningErr(t, r, resumeContent(idB, nameB, "ok-b")); err != nil {
		t.Fatalf("answering b: %v", err)
	}
	if a.runs.Load() != 2 || b.runs.Load() != 2 {
		t.Errorf("runs: approveA=%d approveB=%d, want 2 and 2", a.runs.Load(), b.runs.Load())
	}
}

// A resume that fails must stay retryable with the same reply.
func TestResumeRegression_RetryAfterFailedResume(t *testing.T) {
	a := newCountingAsker("asker", "q")
	a.failFirstResume = true
	var sinkRuns atomic.Int32
	sink := workflow.NewFunctionNode("sink", func(_ agent.Context, in any) (any, error) {
		sinkRuns.Add(1)
		return in, nil
	}, workflow.NodeConfig{})
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, a, sink))

	t1, _ := runReturningErr(t, r, userText("start"))
	id, name := findLongRunningInterrupt(t1)
	if _, err := runReturningErr(t, r, resumeContent(id, name, "ok")); err == nil {
		t.Fatal("first resume: want the injected failure")
	}
	if _, err := runReturningErr(t, r, resumeContent(id, name, "ok")); err != nil {
		t.Fatalf("retry with the same reply: %v", err)
	}
	if sinkRuns.Load() != 1 {
		t.Errorf("sink runs = %d, want 1", sinkRuns.Load())
	}
}

// An AgentNode wrapping a non-LlmAgent (here a sequential agent) whose
// inner LlmAgent asks for tool confirmation. Approving must run the tool.
func TestResumeRegression_ConfirmationInsideSequentialAgentNode(t *testing.T) {
	var toolRuns atomic.Int32
	confirmTool, err := functiontool.New(functiontool.Config{
		Name: "confirm_action", Description: "x", RequireConfirmation: true,
	}, func(agent.Context, struct{}) (map[string]string, error) {
		toolRuns.Add(1)
		return map[string]string{"result": "executed"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	gated, err := llmagent.New(llmagent.Config{Name: "gated", Model: &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromFunctionCall("confirm_action", map[string]any{}, "model"),
		genai.NewContentFromText("finished", "model"),
	}}, Tools: []tool.Tool{confirmTool}})
	if err != nil {
		t.Fatal(err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{gated}}})
	if err != nil {
		t.Fatal(err)
	}
	node, err := workflow.NewAgentNode(seq, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	wf, err := workflowagent.New(workflowagent.Config{Name: workflowAgentName, Edges: workflow.Chain(workflow.Start, node)})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	r := newNodeTestRunner(t, wf, svc)

	t1, _ := runReturningErr(t, r, userText("start"))
	id, name := findLongRunningInterrupt(t1)
	approve := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: id, Name: name, Response: map[string]any{"confirmed": true},
	}}}}
	if _, err := runReturningErr(t, r, approve); err != nil {
		t.Fatalf("approve: %v", err)
	}
	if toolRuns.Load() != 1 {
		t.Errorf("confirmed tool runs = %d, want 1", toolRuns.Load())
	}
}

// TestResumeRegression_ResumePromptIncludesHistory verifies that the model
// call after a resumed confirmation still sees the original user prompt and
// pending function call, rather than only the tool response.
func TestResumeRegression_ResumePromptIncludesHistory(t *testing.T) {
	m := &resumeRecordingModel{}
	f := newConfirmationGraphFixture(t, m)

	t1, _ := runReturningErr(t, f.runner, userText("start"))
	id, _ := findLongRunningInterrupt(t1)
	if id == "" {
		t.Fatal("turn 1 produced no interrupt")
	}
	if _, err := runReturningErr(t, f.runner, confirmationReply(id, true)); err != nil {
		t.Fatalf("approve: %v", err)
	}
	if len(m.contents) < 2 {
		t.Fatalf("model calls = %d, want at least 2", len(m.contents))
	}

	var sawPrompt, sawCall, sawResponse bool
	for _, content := range m.contents[1] {
		for _, part := range content.Parts {
			if part.Text == "start" {
				sawPrompt = true
			}
			if fc := part.FunctionCall; fc != nil && fc.Name == "confirm_action" {
				sawCall = true
			}
			if part.FunctionResponse != nil {
				sawResponse = true
			}
		}
	}
	if !sawPrompt || !sawCall || !sawResponse {
		for i, content := range m.contents[1] {
			t.Logf("content[%d] role=%q", i, content.Role)
			for j, part := range content.Parts {
				t.Logf("  part[%d]=%+v fc=%+v fr=%+v", j, part, part.FunctionCall, part.FunctionResponse)
			}
		}
		t.Errorf("resume prompt missing required history: prompt=%v call=%v response=%v", sawPrompt, sawCall, sawResponse)
	}
}

// TestResumeRegression_ConsumedHandoffReplyRejected verifies that a different
// payload for an already-consumed handoff interrupt is still reported as a
// stale resume rather than silently accepted.
func TestResumeRegression_ConsumedHandoffReplyRejected(t *testing.T) {
	asker := newHitlAsker("asker", "input-1", false)
	sink := workflow.NewFunctionNode("sink", func(agent.Context, any) (any, error) {
		return "done", nil
	}, workflow.NodeConfig{})
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, asker, sink))

	t1, _ := runReturningErr(t, r, userText("start"))
	id, name := findLongRunningInterrupt(t1)
	if id == "" {
		t.Fatal("first turn produced no interrupt")
	}
	if _, err := runReturningErr(t, r, resumeContent(id, name, "yes")); err != nil {
		t.Fatalf("first resume: %v", err)
	}
	_, err := runReturningErr(t, r, resumeContent(id, name, "no"))
	if !errors.Is(err, workflow.ErrNothingToResume) {
		t.Fatalf("second resume error = %v, want %v", err, workflow.ErrNothingToResume)
	}
}

// TestResumeRegression_RootLlmAgentDuplicateReplyStartsFreshRun pins the
// pre-existing root-agent behavior: the graph-specific resume bookkeeping must
// not turn a duplicate reply outside a graph into ErrNothingToResume.
func TestResumeRegression_RootLlmAgentDuplicateReplyStartsFreshRun(t *testing.T) {
	m := &resumeRecordingModel{}
	confirmTool, err := functiontool.New(functiontool.Config{
		Name:                "confirm_action",
		Description:         "performs an action after confirmation",
		RequireConfirmation: true,
	}, func(agent.Context, struct{}) (map[string]string, error) {
		return map[string]string{"result": "executed"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:  "root",
		Model: m,
		Tools: []tool.Tool{confirmTool},
	})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	r := newNodeTestRunner(t, root, svc)

	t1, _ := runReturningErr(t, r, userText("start"))
	id, _ := findLongRunningInterrupt(t1)
	if id == "" {
		t.Fatal("first turn produced no interrupt")
	}
	reply := confirmationReply(id, true)
	if _, err := runReturningErr(t, r, reply); err != nil {
		t.Fatalf("first resume: %v", err)
	}
	callsAfterResume := len(m.contents)
	if _, err := runReturningErr(t, r, reply); err != nil {
		t.Fatalf("duplicate reply: %v", err)
	}
	if got := len(m.contents); got <= callsAfterResume {
		t.Errorf("model calls after duplicate = %d, want more than %d", got, callsAfterResume)
	}
}

type resumeRecordingModel struct {
	contents [][]*genai.Content
}

func (m *resumeRecordingModel) Name() string { return "resume-recording" }

func (m *resumeRecordingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	contents := append([]*genai.Content(nil), req.Contents...)
	m.contents = append(m.contents, contents)
	call := len(m.contents) - 1
	return func(yield func(*model.LLMResponse, error) bool) {
		var content *genai.Content
		if call == 0 {
			content = genai.NewContentFromFunctionCall("confirm_action", map[string]any{}, "model")
		} else {
			content = genai.NewContentFromText("resume complete", "model")
		}
		yield(&model.LLMResponse{Content: content}, nil)
	}
}
