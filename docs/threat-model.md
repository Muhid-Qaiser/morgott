# POC threat model

## Security claim

Within the simulated reference agent, every side-effecting tool request passes
through a fail-closed policy check. A compromised planner cannot invoke an
ungranted tool, alter constrained arguments, or transmit data marked sensitive.

The channel-scoped detectors are shadow-mode sensors. Untrusted content runs
both the indirect sensor and the direct-override fallback. Their scores neither
grant authority nor block a user.

## Boundary

The trusted computing base is the policy document, the `authorize` function,
runtime-supplied provenance and sensitive-data labels, and the code that refuses
to execute denied actions. The model/planner, retrieved documents, email, tool
results, memory, and detector are untrusted. The planner cannot set its own
provenance or clear a sensitive-data label.

The POC uses simulated tool commits; it has no real credentials, network egress,
wallet, email connector, or model API.

The simulation assumes the caller supplies trusted policy/context objects. It
does not bind capabilities to task or user identity, issue/expire credentials,
or validate provenance beyond a nonempty list of strings; provenance does not
yet affect authorization. Those are required runtime integrations, not current
test evidence.

## Assets, actors, sources, and sinks

| Kind | Initial set |
|---|---|
| Assets | tenant data, canary secrets, durable memory, funds, external identity |
| Actors | legitimate user, malicious user, hostile content author, compromised tool |
| Untrusted sources | direct prompt, email, RAG document, web page, tool result, memory |
| Sinks | model response, email, file, durable memory, transaction, external API |

## Invariants

1. Untrusted content may provide facts but cannot grant authority.
2. Every side effect is mediated outside the planner.
3. Sensitive data flows only when the task capability explicitly permits it.
4. Tool output and memory cannot increase privilege.
5. No ambient or long-lived credential is exposed to the planner.
6. Summarization does not erase provenance.
7. Model output is untrusted application input.
8. Human approval, when added, must bind the exact action and arguments.
9. Missing provenance or malformed action schemas fail closed.
10. Detector scores can reduce privilege or escalate review, never authorize.

## Initial abuse cases

| Abuse case | POC control | Status |
|---|---|---|
| Direct user jailbreak | P0 text sensor | measured, bypassable |
| Direct prompt override | P0 text sensor | measured, bypassable |
| Obfuscated/cipher jailbreak | out-of-source holdout | measured, bypassable |
| Human prompt hijacking/extraction | Tensor Trust locked holdout | measured, bypassable |
| Benign or harmful non-injection prompt alerted | OASST1, XSTest, HarmBench, Do-Not-Answer | measured |
| Indirect injection in external content | BIPIA train/test sensor plus reference monitor | measured, bypassable |
| Synthetic agentic tool-output injection | Nemotron transfer audit plus reference monitor | detector 0/676; four representative actions blocked |
| False positive locks out a user | shadow-only allow decision | prevented by design |
| Retrieved text sends a canary | capability plus sensitive-data check | blocked |
| Retrieved text changes email recipient | exact recipient constraint | blocked |
| Retrieved text writes durable memory | tool not granted | blocked |
| Retrieved text transfers funds | tool not granted | blocked |
| Malformed action smuggles fields | exact schema | blocked |
| Authorized safe summary/email | scoped capability | allowed |

## Deferred

AgentDojo/stateful target integration, egress taint across transformations,
credential brokering, exact human approvals, memory quarantine, adaptive
attacks, multilingual attack positives, fuller multi-seed encoder training, and
production-traffic calibration are outside this initial POC. Frozen encoders,
off-the-shelf guard checkpoints, a bounded provider reviewer, and a one-seed
ModernBERT/DeBERTa screen were measured only as shadow experiments.
