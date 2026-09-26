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

package workflow

import (
	"slices"
	"strings"
)

// graph is the precomputed structural view of a workflow's edges.
// Built once at workflow construction; queried by the engine at
// dispatch time.
type graph struct {
	successors    map[Node][]Edge
	predecessors  map[Node][]Edge
	sorted        []Node
	isRootWrapper bool
}

// newGraph indexes edges by source and target node so successor
// and predecessor lookups are both O(1) at dispatch time. The
// returned graph references the input edges by value; mutating the
// input edge slice afterwards does not affect the graph.
func newGraph(edges []Edge) *graph {
	succ := make(map[Node][]Edge)
	pred := make(map[Node][]Edge)
	for _, edge := range edges {
		succ[edge.From] = append(succ[edge.From], edge)
		pred[edge.To] = append(pred[edge.To], edge)
	}
	g := &graph{successors: succ, predecessors: pred}
	// Clipped so cap == len: sortedNodes hands this slice out, and an
	// append by a caller would otherwise write into the graph's array.
	g.sorted = slices.Clip(g.allNodes())
	slices.SortFunc(g.sorted, func(a, b Node) int { return strings.Compare(a.Name(), b.Name()) })
	return g
}

// allEdges returns all edges in the graph, grouped by source node in
// name order.
func (g *graph) allEdges() []Edge {
	var edges []Edge
	for _, n := range g.sortedNodes() {
		edges = append(edges, g.successors[n]...)
	}
	return edges
}

// sortedNodes returns all nodes ordered by name, computed once in
// newGraph. Callers that report findings to the user (graph validation)
// use it so the output does not inherit Go's randomized map iteration
// order. The returned slice is owned by the graph and must not be
// mutated by callers.
func (g *graph) sortedNodes() []Node {
	return g.sorted
}

// unconditionalSCCs labels each node with the strongly connected
// component it belongs to in the subgraph of unconditional
// (Route == nil) edges, by Tarjan's algorithm. Two nodes share a label
// exactly when each is reachable from the other without passing a
// route, which is what makes an edge between them a loop-back rather
// than a branch. A node with no unconditional cycle through it gets a
// label of its own.
func (g *graph) unconditionalSCCs() map[Node]int {
	var (
		index    = make(map[Node]int, len(g.sorted))
		low      = make(map[Node]int, len(g.sorted))
		onStack  = make(map[Node]bool, len(g.sorted))
		comp     = make(map[Node]int, len(g.sorted))
		stack    []Node
		next     int
		nextComp int
	)

	var visit func(n Node)
	visit = func(n Node) {
		index[n], low[n] = next, next
		next++
		stack = append(stack, n)
		onStack[n] = true

		for _, edge := range g.successorsOf(n) {
			if edge.Route != nil {
				continue
			}
			if _, seen := index[edge.To]; !seen {
				visit(edge.To)
				low[n] = min(low[n], low[edge.To])
			} else if onStack[edge.To] {
				low[n] = min(low[n], index[edge.To])
			}
		}

		if low[n] != index[n] {
			return
		}
		for {
			m := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			onStack[m] = false
			comp[m] = nextComp
			if m == n {
				break
			}
		}
		nextComp++
	}

	for _, n := range g.sortedNodes() {
		if _, seen := index[n]; !seen {
			visit(n)
		}
	}
	return comp
}

// successorsOf returns the outgoing edges for a node.
// Returns nil if n has no outgoing edges
// (including the case where n is not in the graph at all). The
// returned slice is owned by the graph and must not be mutated by
// callers.
func (g *graph) successorsOf(n Node) []Edge {
	return g.successors[n]
}

// predecessorsOf returns the incoming edges for a node. Returns
// nil if n has no incoming edges (including the case where n is
// not in the graph at all). The returned slice is owned by the
// graph and must not be mutated by callers.
func (g *graph) predecessorsOf(n Node) []Edge {
	return g.predecessors[n]
}

// allNodes returns all nodes in the graph.
func (g *graph) allNodes() []Node {
	nodes := make(map[Node]bool)
	for n := range g.successors {
		nodes[n] = true
	}
	for n := range g.predecessors {
		nodes[n] = true
	}
	var res []Node
	for n := range nodes {
		res = append(res, n)
	}
	return res
}

// terminalNodeNames returns the names of terminal nodes: those with no
// outgoing edges, excluding the Start sentinel.
func (g *graph) terminalNodeNames() map[string]bool {
	sources := make(map[string]bool)
	for n := range g.successors {
		sources[n.Name()] = true
	}
	terminals := make(map[string]bool)
	for _, n := range g.allNodes() {
		if n.Name() == Start.Name() || sources[n.Name()] {
			continue
		}
		terminals[n.Name()] = true
	}
	return terminals
}
