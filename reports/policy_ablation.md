# Reference-monitor policy ablation

Generated: 2026-07-20T06:57:55+00:00

This is a deterministic simulation of a compromised planner. It tests the authorization boundary, not an LLM and not detector accuracy.

| Guard | Unauthorized committed | Benign committed |
|---|---:|---:|
| Input filter only | 8/8 | 2/2 |
| Reference monitor | 0/8 | 2/2 |

The planner cannot manufacture tool authority, change a constrained recipient, or transmit data that the trusted runtime marks sensitive. Untrusted content may still supply facts for an already authorized summary.

Four attack shapes reference safe categorical metadata from the pinned Nemotron Agentic IPI source; no source environment, identity, prompt, or target arguments are copied into this simulation.
