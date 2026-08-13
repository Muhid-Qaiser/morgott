#!/usr/bin/env bash
# Set up a NON-ROOT user for Claude Code Desktop remote sessions.
#
# Why this exists: the desktop app decides the permission mode before it launches
# the CLI, and it refuses to request Bypass Permissions when the remote account is
# uid 0 -- it silently starts the session in Accept Edits instead. IS_SANDBOX=1
# does NOT help, because that flag is only consulted inside the CLI process, which
# the app has already downgraded via `--permission-mode acceptEdits` on the command
# line. The only way to get bypass mode is to connect as a non-root user.
#
# Idempotent. Run standalone or via bootstrap.sh.
#
#   bash /workspace/setup-remote-user.sh
#
# Afterwards, point the desktop app's SSH connection at  ubuntu@<pod>  instead of
# root@<pod>. The pod itself is still the security boundary; ubuntu gets NOPASSWD
# sudo, so this buys no isolation -- it only satisfies the app's uid check.

set -euo pipefail

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "ERROR: setup-remote-user.sh must run as root." >&2
  exit 1
fi

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
umask 022

RUSER=${RUSER:-ubuntu}
PERSIST=/workspace/home
RHOME=$(getent passwd "$RUSER" | cut -d: -f6 || true)
MIGRATION_STAMP=$(date -u +%Y%m%dT%H%M%SZ)

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m  !!\033[0m %s\n' "$*" >&2; }

# ------------------------------------------------------------ 1. the account
if [ -z "$RHOME" ]; then
  log "Creating user $RUSER"
  useradd -m -s /bin/bash "$RUSER"
  RHOME=$(getent passwd "$RUSER" | cut -d: -f6)
fi
[ -d "$RHOME" ] || { mkdir -p "$RHOME"; chown "$RUSER:$RUSER" "$RHOME"; }
ok "$RUSER -> $RHOME (uid $(id -u "$RUSER"))"

# ------------------------------------------------------------ 2. ssh access
# Same key the pod already accepts for root, so the desktop app needs no new key.
# RunPod re-injects $PUBLIC_KEY into /root/.ssh/authorized_keys on every pod
# start, but only for root -- $RUSER's home is on the container disk and comes
# back empty. Keep a copy on the volume so this still works if we run before
# start.sh has done its injection (e.g. from a pre-start hook).
log "Authorizing SSH keys"
install -d -m 700 -o "$RUSER" -g "$RUSER" "$RHOME/.ssh"
KEYSRC=""
if [ -s /root/.ssh/authorized_keys ]; then
  KEYSRC=/root/.ssh/authorized_keys
  install -m 600 /root/.ssh/authorized_keys "$PERSIST/authorized_keys"   # refresh cache
elif [ -s "$PERSIST/authorized_keys" ]; then
  KEYSRC="$PERSIST/authorized_keys"
  warn "root has no authorized_keys yet -- using the copy cached on the volume"
fi
if [ -n "$KEYSRC" ]; then
  install -m 600 -o "$RUSER" -g "$RUSER" "$KEYSRC" "$RHOME/.ssh/authorized_keys"
  ok "$(wc -l < "$RHOME/.ssh/authorized_keys") key(s) from $KEYSRC"
else
  warn "no authorized_keys anywhere -- $RUSER will have no way to log in"
fi

# ------------------------------------------------------------ 3. shared config
# Symlink onto the network volume so $RUSER and root share one Claude Code state
# dir: settings, memory, project history, and the remote daemon's cached CLI.
# The volume is mounted world-writable, so cross-ownership is not a problem.
# Same mapping root already uses (note tmux.conf is stored undotted).
log "Linking persistent config"
for pair in .agents:.agents .claude:.claude .codex:.codex .config:.config \
            .gitconfig:.gitconfig .tmux.conf:tmux.conf; do
  item=${pair%%:*}; src="$PERSIST/${pair##*:}"
  [ -e "$src" ] || continue
  if [ -L "$RHOME/$item" ]; then
    unlink "$RHOME/$item"
  elif [ -e "$RHOME/$item" ]; then
    quarantine="$RHOME/${item}.bootstrap-migrated-$MIGRATION_STAMP"
    [ ! -e "$quarantine" ] || {
      echo "ERROR: migration quarantine already exists: $quarantine" >&2
      exit 1
    }
    mv "$RHOME/$item" "$quarantine"
    warn "preserved replaced config at $quarantine"
  fi
  ln -s "$src" "$RHOME/$item"
  chown -h "$RUSER:$RUSER" "$RHOME/$item"
  ok "$item -> $src"
done

# .claude.json is per-user mutable state (folder trust, onboarding flags), not
# shared -- copy it so $RUSER does not re-answer the trust prompt on first run.
if [ -f /root/.claude.json ] && [ ! -f "$RHOME/.claude.json" ]; then
  install -m 600 -o "$RUSER" -g "$RUSER" /root/.claude.json "$RHOME/.claude.json"
  ok ".claude.json seeded from root"
fi

# ------------------------------------------------------------ 4. shell env
# env.sh must load above .bashrc's `[ -z "$PS1" ] && return` guard: sshd starts
# the remote daemon with a non-interactive bash, and that daemon's environment is
# what every session it spawns inherits.
log "Configuring shell"
BRC="$RHOME/.bashrc"
touch "$BRC"
ENVLINE="[ -f $PERSIST/env.sh ] && . $PERSIST/env.sh"
MARK_A='# >>> workspace env (setup-remote-user.sh) >>>'
MARK_B='# <<< workspace env (setup-remote-user.sh) <<<'
awk -v env_line="$ENVLINE" -v source_line="source $PERSIST/bashrc.sh" \
    '$0 != env_line && $0 != source_line { print }' \
    "$BRC" > "$BRC.tmp"
mv "$BRC.tmp" "$BRC"
# Prepend above .bashrc's `[ -z "$PS1" ] && return` guard, marker-delimited so a
# re-run replaces the block rather than stacking another copy.
sed -i "\|^$MARK_A\$|,\|^$MARK_B\$|d" "$BRC"
{ printf '%s\n' "$MARK_A" "$ENVLINE" "$MARK_B"; cat "$BRC"; } > "$BRC.tmp"
mv "$BRC.tmp" "$BRC"
echo "source $PERSIST/bashrc.sh" >> "$BRC"
chown "$RUSER:$RUSER" "$BRC"
ok "env.sh loads for non-interactive shells too"

# Ubuntu's stock ~/.profile sources .bashrc first, then prepends $HOME/.local/bin
# to PATH -- which puts the real claude binary ahead of the --yolo wrapper in
# $PERSIST/bin. Re-source env.sh at the very end to restore the intended order.
PRF="$RHOME/.profile"
PMARK_A='# >>> workspace PATH order (setup-remote-user.sh) >>>'
PMARK_B='# <<< workspace PATH order (setup-remote-user.sh) <<<'
PCOMMENT="# keep $PERSIST/bin ahead of ~/.local/bin (see env.sh)"
touch "$PRF"
sed -i "\|^$PMARK_A\$|,\|^$PMARK_B\$|d" "$PRF"
# also strip unmarked leftovers written by earlier versions of this script
awk -v env_line="$ENVLINE" -v comment_line="$PCOMMENT" \
    '$0 != env_line && $0 != comment_line { print }' \
    "$PRF" > "$PRF.tmp" && mv "$PRF.tmp" "$PRF"
printf '%s\n' "$PMARK_A" "$PCOMMENT" "$ENVLINE" "$PMARK_B" >> "$PRF"
chown "$RUSER:$RUSER" "$PRF"
ok "wrapper dir stays first on PATH for login shells"

# ------------------------------------------------------------ 5. sudo
log "Granting passwordless sudo"
echo "$RUSER ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-$RUSER"
chmod 440 "/etc/sudoers.d/90-$RUSER"
ok "/etc/sudoers.d/90-$RUSER"

# ------------------------------------------------------------ 6. claude CLI
# /root is mode 700, so $RUSER cannot reach root's install. Copy the binary into
# the same versions/ layout the installer uses, so self-update keeps working.
log "Installing claude for $RUSER"
SRC_BIN=$(readlink -f /root/.local/bin/claude 2>/dev/null || true)
if [ -n "$SRC_BIN" ] && [ -x "$SRC_BIN" ]; then
  VER=$(basename "$SRC_BIN")
  install -d -o "$RUSER" -g "$RUSER" "$RHOME/.local/share/claude/versions" "$RHOME/.local/bin"
  if [ ! -x "$RHOME/.local/share/claude/versions/$VER" ]; then
    install -m 755 -o "$RUSER" -g "$RUSER" "$SRC_BIN" "$RHOME/.local/share/claude/versions/$VER"
  fi
  ln -sfn "$RHOME/.local/share/claude/versions/$VER" "$RHOME/.local/bin/claude"
  chown -h "$RUSER:$RUSER" "$RHOME/.local/bin/claude"
  ok "claude $VER"
else
  warn "no claude install found under /root -- $RUSER will need 'claude' installed separately"
fi
# codex is a global npm install under /usr, already world-readable.
command -v codex >/dev/null 2>&1 && ok "codex $(codex --version 2>/dev/null | tail -1)"

echo
log "Done. Point the desktop app at ${RUSER}@<pod> instead of root@<pod>."
