#!/usr/bin/env bash
# Sync morgott data with Azure Blob Storage (account vulsightdata, container morgott).
# Azure is the source of truth: every local data change ends with a push.
# See data/README.md for the data card.
#
# Usage:
#   scripts/azsync.sh push [--force] [prefix]   mirror local -> Azure (deletes remote strays)
#   scripts/azsync.sh pull [prefix]    Azure -> local, add/update only
#   scripts/azsync.sh pull --mirror    Azure -> local exact mirror (deletes local strays)
#
# Auth: MORGOTT_SAS_URL (env or .env) if set; otherwise azcopy piggybacks on `az login`.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="https://vulsightdata.blob.core.windows.net/morgott"
PREFIXES=(data/sources data/views data/quarantine data/audits data-archive artifacts/models)
AZCOPY=$(command -v azcopy || echo "$HOME/.local/bin/azcopy")

if [[ -z "${MORGOTT_SAS_URL:-}" && -f .env ]]; then
  MORGOTT_SAS_URL=$(sed -n 's/^MORGOTT_SAS_URL=["'\'']\{0,1\}\([^"'\'']*\)["'\'']\{0,1\}$/\1/p' .env)
fi
if [[ -n "${MORGOTT_SAS_URL:-}" ]]; then
  BASE="${MORGOTT_SAS_URL%%\?*}" SAS="?${MORGOTT_SAS_URL#*\?}"
else
  SAS=""
  export AZCOPY_AUTO_LOGIN_TYPE=AZCLI
fi

remote() { echo "$BASE/$1$SAS"; }

cmd="${1:-}" arg="${2:-}"
case "$cmd" in
  push)
    force=false; [[ "$arg" == "--force" ]] && force=true arg="${3:-}"
    matched=false
    for p in "${PREFIXES[@]}"; do
      [[ -n "$arg" && "$p" != "$arg" ]] && continue
      matched=true
      # ponytail: 50%-count heuristic so a partial local copy can't mirror-delete Azure; tighten if it ever misfires
      if ! $force; then
        rc=$("$AZCOPY" list "$(remote "$p")" 2>/dev/null | grep -c 'Content Length' || true)
        lc=$(find "$p" -type f 2>/dev/null | wc -l)
        if (( rc > 0 && lc * 2 < rc )); then
          echo "!! $p: local has $lc files, Azure has $rc — partial local copy? 'push --force' to mirror anyway" >&2
          exit 1
        fi
      fi
      echo "== push $p"
      "$AZCOPY" sync "$p" "$(remote "$p")" --delete-destination=true --output-level essential
    done
    $matched || { echo "unknown prefix: $arg (one of: ${PREFIXES[*]})" >&2; exit 1; }
    "$AZCOPY" copy data/manifest.json "$(remote data/manifest.json)" --output-level essential
    "$AZCOPY" copy data/README.md "$(remote README.md)" --output-level essential
    ;;
  pull)
    mirror=false; [[ "$arg" == "--mirror" ]] && mirror=true arg=""
    matched=false
    for p in "${PREFIXES[@]}"; do
      [[ -n "$arg" && "$p" != "$arg" ]] && continue
      matched=true
      echo "== pull $p"
      mkdir -p "$p"
      "$AZCOPY" sync "$(remote "$p")" "$p" --delete-destination=$mirror --mirror-mode=$mirror --output-level essential
    done
    $matched || { echo "unknown prefix: $arg (one of: ${PREFIXES[*]})" >&2; exit 1; }
    # manifest.json and data/README.md come from git, not pulled.
    ;;
  *)
    sed -n '2,11p' "$0"; exit 1 ;;
esac
