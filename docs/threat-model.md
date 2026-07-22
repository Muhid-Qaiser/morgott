# Threat model

## Security claim

morgott makes a narrow architectural claim: a compromised planner must not gain
authority from text. In the current simulation, every side-effecting proposal
passes through a fail-closed reference monitor with caller-supplied capabilities
and exact argument constraints.

The legacy injection-control classifier and corpus routing labels are predictive
only; no routing classifier has been trained. Neither current labels nor future
scores may block a user or approve a tool action. Finance, cybersecurity, and
other sensitive topics are not deny rules; exact side effects require scoped
authority.

## Trust boundary

Trusted:

- static policy and reference-monitor code;
- runtime-supplied capability, provenance, and sensitive-data metadata;
- the code that refuses to execute denied actions.

Untrusted:

- users, models, planners, retrieved documents, email, web pages, tool output,
  memory, summaries, classifier scores, and generated labels.

The POC uses simulated commits. It has no real credentials, wallet, email
connector, network egress, or model API. It does not yet bind capabilities to a
task/user identity, issue expirations, or propagate provenance through a live
agent runtime.

## Assets and sinks

| Kind | Examples |
|---|---|
| Assets | tenant data, secrets, durable memory, funds, external identity |
| Actors | legitimate user, malicious user, hostile content author, compromised tool |
| Untrusted sources | prompt, email, RAG document, web page, tool result, memory |
| Sinks | response, email, file, memory, transaction, external API |

## Invariants

1. Untrusted content may provide facts but cannot grant authority.
2. Every side effect is mediated outside the planner.
3. Sensitive data flows only when an exact capability permits it.
4. Tool output, memory, summaries, and detector scores cannot increase privilege.
5. Credentials are not exposed to the planner.
6. Model output is untrusted application input.
7. Future human approval binds the exact action and arguments.
8. Missing provenance and malformed action schemas fail closed.
9. `uncertain` never becomes benign by default.
10. Train, validation, and dev-test groups and exact texts remain disjoint;
    conflicts and leakage stay visible in quarantine.

## Abuse cases

| Abuse case | Required control |
|---|---|
| Direct jailbreak or prompt override | advisory detector plus deterministic action policy |
| Injection in email, RAG, web, tool output, or memory | provenance-aware routing plus the same action policy |
| Obfuscated or unseen attack bypasses the detector | reference monitor still denies ungranted side effects |
| Benign security/finance discussion is flagged | shadow-only routing and measured review cost |
| Harmful content is confused with injection | independent label axes and masked subtype supervision |
| Source ambiguity becomes a benign label | nullable labels and separate uncertain/auxiliary roles |
| Train/evaluation leakage inflates results | grouped splits, exact blocking, near-overlap quarantine |
| Retrieved text changes a recipient or transfers funds | exact tool and argument constraints |
| Retrieved text exfiltrates a secret | sensitive-data egress policy |
| Retrieved text writes durable memory | capability denial or quarantined memory writes |

## Deferred

Stateful agent integration, taint propagation, credential brokering, exact human
approval, durable-memory quarantine, adaptive attacks, multimodal/browser
injection, and production-traffic calibration are future work. No text detector
is a substitute for those controls.
