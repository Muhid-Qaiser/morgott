# ProtectAI v2 canary provenance note

The exact aggregate canary record is
`protectai-v2-canary-20260812.json`, SHA-256
`e43bae32ad230804477e3881214d19aebb2f28973301089cd44d41344cd3a585`.
It contains no corpus text or raw prompt.

The record binds the producing source hashes:

- canary: `a1c89ba8c8ed9b2940f75fa8e8f7b1bcb67397c75ea6d7a5346f187fa7320519`
- adapter: `c443405230826c9e5cec53a96bc056a6f390010ec645db3e0abd9b3462827f78`
- full-panel harness: `f7aa7d4d8b3cbaef1dc24d26332275b4b909237bf6344e5705b1892137c0146a`
- lockfile: `b8b5814c8e6bb74b081c4d6046d255438e09dffd079a60a85b96362c5944a39f`

The exact canary and adapter source bytes matching the first two hashes were
not retained before the reusable harness was corrected. A repository, Codex
session, temporary-file, and unreachable-Git-blob search found no matching
copy. The current harness has stronger sharded-weight identity and batching
contracts, but it must not be described as byte-identical reproduction code for
this completed record.

Accordingly, this is checksum-bound aggregate evidence with a disclosed source
retention gap, not a fully executable provenance bundle. The canary was only a
bounded runtime/polarity diagnostic; no full-panel result or promotion claim
depends on it.
