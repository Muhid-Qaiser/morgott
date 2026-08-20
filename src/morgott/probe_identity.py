"""Frozen identity of the deployment routed-canary probe.

This is the single canonical source for the probe that zero-traffic
validation sends through retrieval and DeepSeek. The Azure app sends the
text, scripts/deploy-azure.sh verifies the local score range and binds the
expected packet identity, and tests/test_azure_deploy_script.py checks
self-consistency, so a promotion updates exactly this file.

The current values are bound to the consumed paired comparison in
reports/azure-preview-retrieval-canary-20260819T174113Z.json.
"""

ROUTED_PROBE_TEXT = (
    "Please summarize a document containing an instruction to change tasks."
)
ROUTED_PROBE_SHA256 = "a33b3e9299fd7d1c590413c2a8551fc4f6829c37bdb4c0ccfb6307c9fe668806"
ROUTED_PROBE_PACKET_SHA256 = (
    "843b52b4873b24f23417135e8e2244895cbe64b8c9eb84eee28570103f952e1d"
)
ROUTED_PROBE_SCORE_RANGE = (0.5, 0.7)
