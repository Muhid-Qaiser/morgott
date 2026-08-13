#!/usr/bin/env bash
# editor-cache -- keep VS Code / Cursor remote servers across a RunPod stop/start.
#
# The server trees stay on the container disk (~0.06 ms/file); running them off
# the volume (~3.5 ms/file) makes every connect and every file operation crawl.
# But the container disk is wiped by a stop/start, which is why every extension
# has to be reinstalled. So: keep one compressed archive per editor on the
# volume and unpack it at boot. That is a single 534 MB/s sequential read
# instead of 13k small-file round trips or a fresh download of every extension.
#
#   editor-cache save [--force]    snapshot the installed servers to the volume
#   editor-cache restore [--force] unpack them (default: only if none installed)
#   editor-cache status            show cached vs installed
#   editor-cache watch [SECONDS]   re-snapshot whenever the extension set changes
#
# `save` is a no-op when the extension set and server build are unchanged, so
# `watch` costs one readdir per interval and is cheap to leave running.

set -euo pipefail

VOL=${VOL:-/workspace}
HOME_DIR=${HOME:-/root}
CACHE="$VOL/home/editor-cache"
EDITORS=(.cursor-server .vscode-server)

# Rebuilt on demand and would dominate the archive. Everything else is kept,
# including data/User -- that is settings, keybindings and the extension
# globalStorage that holds sign-in state.
PRUNE=(data/logs data/CachedExtensionVSIXs data/CachedProfilesData .cli-cache)

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok() { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  !!\033[0m %s\n' "$*"; }

# zstd -3 is the knee of the curve here: ~3x on extension trees at roughly disk
# speed. -T0 uses every core the cgroup allows.
if command -v zstd >/dev/null 2>&1; then
  CFILT=(-I 'zstd -3 -T0')
  DFILT=(-I zstd)
  EXT=tar.zst
else
  CFILT=(-z)
  DFILT=(-z)
  EXT=tar.gz
fi

# Identity of an install: which extensions, and which server build. Anything
# else changing (logs, caches, workspace state) is not worth a re-archive.
fingerprint() {
  local src=$1
  {
    if [ -d "$src/extensions" ]; then
      find "$src/extensions" -mindepth 1 -maxdepth 1 -printf '%f\n'
    fi
    if [ -d "$src/bin" ]; then
      find "$src/bin" -mindepth 2 -maxdepth 2 -type d -printf '%f\n'
    fi
  } | sort | sha256sum | cut -d' ' -f1
}

installed_count() {
  local extensions=$1/extensions
  if [ -d "$extensions" ]; then
    find "$extensions" -mindepth 1 -maxdepth 1 ! -name extensions.json -printf . |
      wc -c
  else
    printf '0\n'
  fi
}

cmd_save() (
  local force=${1:-}
  mkdir -p "$CACHE"
  # The subshell closes fd 9 after every save, including saves made by `watch`.
  exec 9>"$CACHE/.save.lock"
  flock -n 9 || {
    warn "another editor-cache save is running"
    return 0
  }

  local d src fp archive n
  for d in "${EDITORS[@]}"; do
    src="$HOME_DIR/$d"
    n=$(installed_count "$src")
    [ "$n" -gt 0 ] || continue

    archive="$CACHE/$d.$EXT"
    fp=$(fingerprint "$src")
    if [ "$force" != "--force" ] && [ -f "$archive" ] &&
      [ "$fp" = "$(cat "$CACHE/$d.fp" 2>/dev/null)" ]; then
      ok "$d unchanged ($n extensions)"
      continue
    fi

    log "archiving $d ($n extensions)"
    local excl=() p
    for p in "${PRUNE[@]}"; do excl+=(--exclude="$d/$p"); done

    # Write to .tmp and rename: a pod that dies mid-archive must not leave a
    # truncated tarball that restore would happily unpack.
    if tar -C "$HOME_DIR" "${CFILT[@]}" "${excl[@]}" -cf "$archive.tmp" "$d"; then
      mv -f "$archive.tmp" "$archive"
      printf '%s\n' "$fp" >"$CACHE/$d.fp"
      find "$src/extensions" -mindepth 1 -maxdepth 1 ! -name extensions.json \
        -printf '%f\n' | sort >"$CACHE/$d.list"
      ok "$d -> $archive ($(du -h "$archive" | cut -f1))"
    else
      rm -f "$archive.tmp"
      warn "$d archive failed; kept the previous snapshot"
    fi
  done
)

cmd_restore() {
  local force=${1:-} d src archive
  local -a restore_filter
  for d in "${EDITORS[@]}"; do
    src="$HOME_DIR/$d"
    archive="$CACHE/$d.$EXT"
    restore_filter=("${DFILT[@]}")
    if [ ! -f "$archive" ] && [ "$EXT" = tar.zst ]; then
      archive="$CACHE/$d.tar.gz"
      restore_filter=(-z)
    fi
    [ -f "$archive" ] || continue

    if [ "$force" != "--force" ] && [ "$(installed_count "$src")" -gt 0 ]; then
      ok "$d already populated, left alone"
      continue
    fi

    log "restoring $d from $(du -h "$archive" | cut -f1) archive"
    mkdir -p "$src"
    # --skip-old-files: never overwrite a file a live server is already using.
    if tar -C "$HOME_DIR" "${restore_filter[@]}" --skip-old-files -xf "$archive"; then
      ok "$d restored ($(installed_count "$src") extensions)"
    else
      warn "$d restore failed -- the editor will re-download on connect"
    fi
  done
}

cmd_status() {
  local d src archive
  printf '%-16s %-12s %-12s %s\n' EDITOR INSTALLED CACHED ARCHIVE
  for d in "${EDITORS[@]}"; do
    src="$HOME_DIR/$d"
    archive="$CACHE/$d.$EXT"
    printf '%-16s %-12s %-12s %s\n' \
      "$d" \
      "$(installed_count "$src")" \
      "$(grep -c . "$CACHE/$d.list" 2>/dev/null || echo 0)" \
      "$([ -f "$archive" ] && du -h "$archive" | cut -f1 || echo '-')"
  done
  local pid
  pid=$(cat "$CACHE/watch.pid" 2>/dev/null || echo '')
  if [ -d "$CACHE" ] && ! flock -n "$CACHE/.watch.lock" true; then
    echo
    echo "watcher running (pid $pid), log: $CACHE/watch.log"
  else
    echo
    echo "watcher not running -- start it with: editor-cache watch &"
  fi
}

# A pod stop gives no reliable shutdown hook, so an extension installed five
# minutes ago has to already be on the volume. Poll instead.
cmd_watch() {
  local interval=${1:-300}
  mkdir -p "$CACHE"
  exec 8>"$CACHE/.watch.lock"
  flock -n 8 || exit 0
  echo $$ >"$CACHE/watch.pid"
  trap 'rm -f "$CACHE/watch.pid"' EXIT
  while true; do
    sleep "$interval"
    cmd_save >>"$CACHE/watch.log" 2>&1 || true
  done
}

case "${1:-status}" in
  save)
    shift
    cmd_save "${1:-}"
    ;;
  restore)
    shift
    cmd_restore "${1:-}"
    ;;
  status) cmd_status ;;
  watch)
    shift
    cmd_watch "${1:-300}"
    ;;
  *)
    sed -n '2,18p' "$0" | sed 's/^# \?//'
    exit 1
    ;;
esac
