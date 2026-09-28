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
	"context"
	"net/url"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
)

// This file lives in internal/ rather than in agent/remoteagent/v2 because
// server/adka2a/v2 issues its own CancelTask against a remote subagent's client
// and needs the identical scope, and it cannot import agent/remoteagent/v2 —
// that package imports adka2a.

// CredentialScope is the key a session-aware [auth.CredentialProvider] sees via
// [a2aclient.SessionIDFrom]. It is the whole calling identity, not just the
// session id: the id alone is caller-supplied and optional, so two users each
// holding a session called "default" would otherwise resolve to one credential.
//
// agentName is part of the key because a bearer token is scoped to an audience
// as well as to a user. Without it, one provider shared between two remote
// agents in the same session would hand the second agent the token minted for
// the first.
//
// The parts are percent-encoded and joined with "/", so no two identities can
// produce the same key and the result stays printable. A provider that wants
// the parts back splits on "/" and calls [url.QueryUnescape] on each.
func CredentialScope(appName, userID, sessionID, agentName string) a2aclient.SessionID {
	return a2aclient.SessionID(url.QueryEscape(appName) + "/" + url.QueryEscape(userID) +
		"/" + url.QueryEscape(sessionID) + "/" + url.QueryEscape(agentName))
}

type agentCardKey struct{}

// WithAgentCard carries the resolved card so the credentials adapter can tell
// which kind of security scheme a name refers to. The a2a CredentialsService
// interface hands it only the name.
func WithAgentCard(ctx context.Context, card *a2a.AgentCard) context.Context {
	return context.WithValue(ctx, agentCardKey{}, card)
}

// AgentCardFrom returns the card [WithAgentCard] attached, or nil.
func AgentCardFrom(ctx context.Context) *a2a.AgentCard {
	card, _ := ctx.Value(agentCardKey{}).(*a2a.AgentCard)
	return card
}

// CallIdentity is who is calling whom, for one outgoing A2A call.
type CallIdentity struct {
	AppName   string
	UserID    string
	SessionID string
	// AgentName is the local name of the remote agent being called.
	AgentName string
}

// AttachAuthScope returns ctx carrying what the a2a auth interceptor needs to
// resolve a credential for one outgoing call: the scope, and the card that
// decides where the credential is written. Pass a nil card when it is not
// resolved yet and add it later with [WithAgentCard].
//
// It is a no-op unless cfg.OwnsAuthScope says this SDK put the interceptor
// there. A caller who wired their own interceptor owns their own key, and
// overwriting it would silently drop their credential.
func AttachAuthScope(ctx context.Context, cfg *A2AServerConfig, id CallIdentity, card *a2a.AgentCard) context.Context {
	if !cfg.OwnsAuthScope {
		return ctx
	}
	ctx = a2aclient.AttachSessionID(ctx, CredentialScope(id.AppName, id.UserID, id.SessionID, id.AgentName))
	return WithAgentCard(ctx, card)
}

// CardNamesNoScheme reports whether the card gives the a2a AuthInterceptor
// nothing to ask about, so it never calls Get and the request leaves with no
// credential and nothing logged.
//
// It has to agree with the interceptor exactly, because the two warnings are
// meant to be mutually exclusive. The interceptor bails on a nil requirement
// list or a nil scheme map, then iterates the requirement objects and the
// scheme names inside each. A nil requirement list needs no test of its own
// here — the loop below reaches the same answer by iterating nothing — so what
// is left is the nil scheme map and a requirement object naming nothing. The
// latter is the one a real card carries: security: [{}] is how OpenAPI spells
// "authentication optional".
//
// Nil rather than empty is deliberate and is the interceptor's own test. A card
// whose JSON says "securitySchemes": {} decodes to an empty but non-nil map, so
// the interceptor does ask, finds nothing that can carry the credential, and
// the mismatch warning covers it. Reading that case as "names no scheme" here
// would warn about it twice.
func CardNamesNoScheme(card *a2a.AgentCard) bool {
	if card == nil || card.SecuritySchemes == nil {
		return true
	}
	for _, requirement := range card.SecurityRequirements {
		if len(requirement) > 0 {
			return false
		}
	}
	return true
}
