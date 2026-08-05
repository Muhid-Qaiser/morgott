# Threat model

## Security claim

morgott makes a narrow architectural claim: a compromised planner must not gain
authority from text. In the current simulation, every side-effecting proposal
passes through a fail-closed reference monitor with caller-supplied capabilities
and exact argument constraints.

The legacy injection-control classifier, the word routing baseline, and corpus routing labels are predictive only.
The routing baseline uses the untouched 0.5 cutoff and has neither production
calibration nor a prospectively labelled final test.
Neither current labels nor future scores may block a user or approve a tool
action.
Finance, cybersecurity, and other sensitive topics are not deny rules; exact
side effects require scoped authority.
Some identical text is legitimate when supplied by an authorized user and hostile when embedded in untrusted content.
A text-only classifier cannot recover that missing authority or provenance, so channel and actor identity must come from trusted runtime metadata.

## Trust boundary

Trusted:

- static policy and reference-monitor code;
- runtime-supplied capability, provenance, and sensitive-data metadata;
- the code that refuses to execute denied actions.

Untrusted:

- users, models, planners, retrieved documents, email, web pages, tool output,
  memory, summaries, classifier scores, generated labels, and remote model responses.

The POC uses simulated commits and has no wallet, email connector, or live capability runtime.
The optional shadow cascade makes an OpenRouter model API call only when `--allow-remote` is set and `OPENROUTER_API_KEY` is available.
For remote-enabled multi-window untrusted content without a local high, the complete normalized artifact leaves the process for one full-context review.
A clear full-context result may then send middle-zone windows for the existing fallback review, while direct-user and single-window routing stay unchanged.
The API key remains inside the provider client.
Remote responses are untrusted, strictly parsed, and converted to a conservative incomplete assessment when invalid.
Without `--allow-remote`, no artifact text leaves the maintained cascade.
The completed PredictStrategy evaluation used the same provider boundary and recorded only hashes, parsed values, timings, usage, and cost.
The POC does not yet bind capabilities to a task or user identity, issue expirations, or propagate provenance through a live agent runtime.

## Formal authorization boundary

A model may translate a natural-language task into a proposed typed action, a proposed permission request, or evidence spans.
Those outputs are untrusted application input and may not mint a capability, assign trusted provenance, mark data non-sensitive, or choose a trusted sink identity.
Capability identity, source identity, sensitivity, approval, and policy come from trusted runtime state.
Unknown identity or missing required labels fail closed before a side effect.

A deterministic solver or reference monitor proves only that its supplied graph and policy permit an action.
It cannot prove that an LLM extracted every obligation or labeled every relevant flow.
The [FAVA paper](https://arxiv.org/abs/2607.27267v1) makes this boundary explicit and reports extraction misses among its remaining errors.
Morgott therefore treats model-derived permission structures as proposals that can narrow or request authority, never as the authority source itself.

## Argument provenance contract

Whole-context provenance remains a conservative fallback, not the target runtime shape.
A future runtime capability must be able to constrain a variable argument to an allowed source identity and field, with the caller supplying that argument's complete lineage from trusted instrumentation.
Missing argument lineage, unexpected argument keys, and sources outside the capability-bound set must fail closed.

Every security-critical argument should be constrained either to an exact value or to an allowed source identity and field.
Runtime transformations must preserve the complete source set, and combining values must union their provenance, including sensitivity and producer provenance, rather than selecting the most convenient origin; an omitted producer must be conservatively tagged as untrusted planner output.
A trusted adapter may register an exact resource identity returned by an already-authorized effect for a dependent capability, but it must bind the actual returned field and must never trust a planner-supplied copy of that identity.

This permits facts from an untrusted record to participate in a specifically authorized source-to-sink flow without letting the record add a tool, recipient, amount, or second action.
The planner may propose an attribution, but only trusted runtime instrumentation may establish it.
There is no live agent runtime in this repository, so this remains a documented contract rather than a maintained module, matching the clarification-contract decision: a source-lineage primitive without a real caller would be a shallow module, and the deterministic simulator keeps whole-context provenance.
The [AuthGraph paper](https://arxiv.org/abs/2605.26497v1) supports parameter-source checking as a useful seam, while its same-source poisoning limitation shows why source authorization and content safety remain separate controls.

## Denial feedback contract

A denied action does not end the task by itself.
The trusted runtime may return a stable machine-readable denial code plus a minimal recovery instruction so the planner can continue permitted work.
The feedback must not reveal capability contents, secret values, hidden policy branches, or a list of alternative privileged actions.
It cannot modify the capability, grant an exception, or authorize the retry.
Every revised proposal passes through the same reference monitor as a new action.

The completed Agent-Diff recovery case supports this primitive within one consumed task, and the independent [ActPlane artifact](https://github.com/eunomia-bpf/ActPlane/tree/63db86945c9b8618a46aa68c8de214bc4b8343d9) reports materially higher decision compliance with semantic feedback than with opaque enforcement on two model backbones.
Neither result makes model interpretation part of the security boundary.
The deterministic denial remains the control; feedback is only a utility aid after that control has fired.

## Clarification contract

A clarification reply is data for an already-declared missing slot, not authority to widen the task.
The [ASPI diagnostic](../reports/agent-security-benchmark-options.md#aspi-clarification-state-audit-exposed-a-routing-blind-spot) shows why response-only routing is insufficient: 31.41% of its retained attack replies pass below the direct-user review gate because the text often resembles an ordinary additional request.

The future runtime seam belongs between dialogue handling and capability issuance.
Its clarification module should expose one small interface that accepts a trusted challenge plus an untrusted reply and returns either a typed bound value, rejection, or `new_task_required`.
The trusted challenge must bind an immutable task and capability identity, one slot name, its type and validation constraints, and a bounded use count.
The reply may fill only that slot and must never select a tool, add an action, alter another argument, increase a use count, choose a new sink, or mint a capability.
Any requested scope expansion must leave the clarification flow and enter the ordinary task-authorization or exact-approval flow as a new request.
The resulting proposed action still passes through the reference monitor against the original capability.
Learned routing may reduce exposure or request review, but it cannot bind the slot, approve expansion, or change the capability.

There is no live dialogue runtime in this repository, so adding a speculative clarification implementation would create a shallow module with no real caller.
The contract remains a required interface for that future runtime, and its tests must exercise the interface rather than a model transcript.

## Assets and sinks

| Kind | Examples |
|---|---|
| Assets | tenant data, secrets, OpenRouter API key, durable memory, funds, external identity |
| Actors | legitimate user, malicious user, hostile content author, compromised tool |
| Untrusted sources | prompt, email, RAG document, web page, tool result, memory |
| Sinks | response, email, file, memory, transaction, OpenRouter model API |

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
10. Clarification replies bind only typed slots and cannot expand task authority.
11. Denial feedback cannot reveal or expand authority, and every retry is reauthorized.
12. Model-extracted permission structure can request or narrow authority but cannot grant it.
13. Security-critical arguments retain trusted runtime provenance and match capability-bound source fields.
14. Train, validation, and dev-test groups and exact texts remain disjoint;
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
| Malformed or adversarial provider response changes routing | strict schema and logprob parsing plus conservative incomplete restriction |
| NOOA tracing persists corpus text | refuse configured tracing and never retain raw provider payloads |
| Retrieved text changes a recipient or transfers funds | exact tool and argument constraints |
| Retrieved text exfiltrates a secret | sensitive-data egress policy |
| Retrieved text writes durable memory | capability denial or quarantined memory writes |
| Clarification reply adds a second action | typed slot binding plus a separate new-task authorization flow |
| Denial feedback is used as an override | non-secret feedback plus fresh authorization of every retry |

## Deferred

Stateful agent integration, live taint instrumentation, credential brokering, exact human
approval, durable-memory quarantine, adaptive attacks, multimodal/browser
injection, and production-traffic calibration are future work. No text detector
is a substitute for those controls.
