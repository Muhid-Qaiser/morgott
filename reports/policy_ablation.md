# Reference-monitor policy ablation

This is a deterministic simulation of a compromised planner. It tests the authorization boundary, not an LLM and not detector accuracy.

The comparison holds planner proposals fixed and assumes every proposal commits when no action monitor is present.

| Guard | Unauthorized committed | Benign committed |
|---|---:|---:|
| No action monitor | 9/9 | 3/3 |
| Reference monitor | 0/9 | 3/3 |

The planner cannot manufacture tool authority, change a constrained recipient, or transmit data that the trusted runtime marks sensitive. Untrusted content may still supply facts for an already authorized summary.

Four attack shapes reference safe categorical metadata from the pinned Nemotron Agentic IPI source; no source environment, identity, prompt, or target arguments are copied into this simulation.

One synthetic response-egress scenario is motivated by the pinned Financial AI CTF outcome metadata. No protected value, participant text, system prompt, or model response is copied into the simulation.
