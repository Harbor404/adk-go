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

package remoteagent

import (
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
)

// TestAttachAuthScope covers the gate that decides whether this SDK owns the
// credential scope on an outgoing call. Overwriting a scope a caller attached
// for their own interceptor would resolve under a key their store has never
// seen and drop their credential.
func TestAttachAuthScope(t *testing.T) {
	card := &a2a.AgentCard{Name: "remote"}
	id := CallIdentity{AppName: "shop", UserID: "alice", SessionID: "s1", AgentName: "crm"}

	t.Run("owned", func(t *testing.T) {
		ctx := AttachAuthScope(t.Context(), &A2AServerConfig{OwnsAuthScope: true}, id, card)
		sid, ok := a2aclient.SessionIDFrom(ctx)
		if !ok {
			t.Fatal("no credential scope attached")
		}
		if want := a2aclient.SessionID("shop/alice/s1/crm"); sid != want {
			t.Errorf("scope = %q, want %q", sid, want)
		}
		if AgentCardFrom(ctx) != card {
			t.Error("the agent card was not attached; the credentials adapter reads it to decide placement")
		}
	})

	t.Run("not owned leaves the caller's context alone", func(t *testing.T) {
		theirs := a2aclient.AttachSessionID(t.Context(), "caller-tenant")
		ctx := AttachAuthScope(theirs, &A2AServerConfig{}, id, card)
		sid, _ := a2aclient.SessionIDFrom(ctx)
		if want := a2aclient.SessionID("caller-tenant"); sid != want {
			t.Errorf("scope = %q, want %q; a caller who wired their own interceptor owns the key", sid, want)
		}
		if AgentCardFrom(ctx) != nil {
			t.Error("an agent card was attached to a context this SDK does not own")
		}
	})
}

// TestAgentCardFromEmptyContext pins the default-deny end of the card lookup:
// the credentials adapter treats a missing card as "no scheme can carry this",
// so a nil here must not be confused with a card.
func TestAgentCardFromEmptyContext(t *testing.T) {
	if got := AgentCardFrom(t.Context()); got != nil {
		t.Errorf("AgentCardFrom() = %v, want nil", got)
	}
}
