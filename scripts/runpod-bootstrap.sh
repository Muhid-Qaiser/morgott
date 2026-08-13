#!/usr/bin/env bash
# RunPod pod bootstrap — restores a reset/redeployed pod to a working state.
# Everything durable lives on the network volume at /workspace.
# Idempotent: safe to re-run on an already-configured pod.
#
# Usage:  bash /workspace/bootstrap.sh
#
# Storage policy (measured on this pod, 2026-08-06):
#   volume  /workspace : 534 MB/s sequential, ~3.5 ms per small file
#   container disk /   : 13.4 GB/s sequential, ~0.06 ms per small file
# => big blobs (weights, wheels, datasets) on the volume;
#    small-file-heavy trees (editor servers, node_modules, .local) stay local
#    and get reinstalled by this script.

set -euo pipefail

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "ERROR: bootstrap must run as root." >&2
  exit 1
fi

# Do not inherit a writable persistent PATH while this script has root
# authority. User shell customization is installed later for the non-root
# remote account; it is never sourced by this process.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
umask 022

VOL=/workspace
HOME_DIR=/root
PERSIST="$VOL/home"
MORGOTT_DIR="$VOL/code/morgott"
REMOTE_ACCOUNT=${REMOTE_USER:-ubuntu}
MIGRATION_STAMP=$(date -u +%Y%m%dT%H%M%SZ)

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok() { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  !!\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- 0. sanity
if [ ! -d "$VOL" ]; then
  echo "ERROR: $VOL not found. Attach the network volume at deploy time." >&2
  echo "Network volumes cannot be attached after the pod is created." >&2
  exit 1
fi
log "Volume: $(df -h "$VOL" | awk 'NR==2 {print $4" free of "$2}')"
bootstrap_mode=$(stat -c '%a' "$0")
if (((8#$bootstrap_mode & 0022) != 0)); then
  warn "$0 is group/other-writable (mode $bootstrap_mode); this volume cannot enforce executable trust"
  warn "review its SHA-256 before running it as root on every replacement pod"
fi

# Several restore steps drop privileges to the remote account. Establish that
# identity before the first one; the setup helper configures it fully later.
if ! id "$REMOTE_ACCOUNT" >/dev/null 2>&1; then
  log "Creating remote account $REMOTE_ACCOUNT"
  useradd -m -s /bin/bash "$REMOTE_ACCOUNT"
fi

# ------------------------------------------- 1. persistent state (symlinked)
# Only credential/config dirs go on the volume. They are small, and losing them
# costs a re-login; losing .local just costs a re-install, which is automated.
log "Linking persistent config"
mkdir -p "$PERSIST"/{.agents,.claude,.codex,.config}
mkdir -p "$VOL"/{projects,code,hf_cache,datasets,checkpoints,.cache}

link() {
  local target=$1 name=$2
  mkdir -p "$target"
  if [ -L "$HOME_DIR/$name" ]; then
    # replace any existing symlink, including a dangling one
    unlink "$HOME_DIR/$name"
  elif [ -d "$HOME_DIR/$name" ]; then
    # Copy first, then prove that every source entry reached the destination.
    # Preserve the original as a recoverable quarantine instead of deleting it.
    rsync -a --ignore-existing "$HOME_DIR/$name/" "$target/"
    local remaining quarantine
    remaining=$(rsync -ani --checksum "$HOME_DIR/$name/" "$target/")
    if [ -n "$remaining" ]; then
      echo "ERROR: incomplete migration for $HOME_DIR/$name" >&2
      return 1
    fi
    quarantine="$HOME_DIR/${name}.bootstrap-migrated-$MIGRATION_STAMP"
    [ ! -e "$quarantine" ] || {
      echo "ERROR: migration quarantine already exists: $quarantine" >&2
      return 1
    }
    mv "$HOME_DIR/$name" "$quarantine"
    warn "preserved migrated source at $quarantine"
  elif [ -e "$HOME_DIR/$name" ]; then
    echo "ERROR: refusing to replace non-directory $HOME_DIR/$name" >&2
    return 1
  fi
  ln -sfn "$target" "$HOME_DIR/$name"
}

for d in .agents .claude .codex .config; do link "$PERSIST/$d" "$d"; done
ok "config dirs -> $PERSIST"

# Root creates and migrates these trees, but the remote account owns every
# interactive session and all restored project environments. Fix existing
# volume contents too, then fail before the first dropped-privilege command if
# the mount does not actually permit writes.
mkdir -p "$PERSIST/editor-cache"
remote_group=$(id -gn "$REMOTE_ACCOUNT")
remote_writable=(
  "$PERSIST/.agents"
  "$PERSIST/.claude"
  "$PERSIST/.codex"
  "$PERSIST/.config"
  "$PERSIST/editor-cache"
  "$VOL/.cache"
  "$VOL/checkpoints"
  "$VOL/code"
  "$VOL/datasets"
  "$VOL/hf_cache"
  "$VOL/projects"
)
chown -R "$REMOTE_ACCOUNT:$remote_group" "${remote_writable[@]}"
for d in "${remote_writable[@]}"; do
  sudo -u "$REMOTE_ACCOUNT" test -w "$d" || {
    echo "ERROR: $REMOTE_ACCOUNT cannot write persistent path $d" >&2
    exit 1
  }
done
ok "persistent working trees writable by $REMOTE_ACCOUNT"

# Editor servers deliberately stay on the container disk. Running VS Code /
# Cursor server off MooseFS makes every connect and every file op crawl.
for d in .vscode-server .cursor-server; do
  if [ -L "$HOME_DIR/$d" ]; then rm -f "$HOME_DIR/$d"; fi
  mkdir -p "$HOME_DIR/$d"
done

# The persisted editor archives use zstd when it is available. Install it
# before launching the restore and its long-lived watcher so both processes
# select the same format that section 3 guarantees for the finished pod.
if ! command -v zstd >/dev/null 2>&1; then
  log "Installing zstd for editor-cache restore"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd >/dev/null
fi

# ...which is exactly why a stop/start used to mean reinstalling every
# extension by hand. editor-cache keeps one compressed archive per editor on
# the volume and unpacks it here: a single sequential read at 534 MB/s instead
# of 13k small-file round trips, and no marketplace download at all.
EDITOR_CACHE_SHA256=6969c800cc2034c334c1449e613b0610bdd8e9adb867dd0d817e5fbbd3c0c459
EDITOR_CACHE_HELPER="$MORGOTT_DIR/scripts/runpod-editor-cache.sh"
[ -x "$EDITOR_CACHE_HELPER" ] || EDITOR_CACHE_HELPER="$PERSIST/bin/editor-cache"
if [ -x "$EDITOR_CACHE_HELPER" ] &&
  id "$REMOTE_ACCOUNT" >/dev/null 2>&1 &&
  printf '%s  %s\n' "$EDITOR_CACHE_SHA256" "$EDITOR_CACHE_HELPER" |
  sha256sum -c - >/dev/null 2>&1; then
  # The reviewed helper is hash-pinned and still executes without root authority.
  sudo -u "$REMOTE_ACCOUNT" -H "$EDITOR_CACHE_HELPER" restore ||
    warn "editor-cache restore failed"
  # A pod stop gives no shutdown hook, so an extension installed ten minutes
  # ago has to already be on the volume. Poll instead -- `save` is a no-op
  # unless the installed extension set or the server build actually changed.
  # setsid + </dev/null so it outlives this script's process group; the script
  # itself refuses to start a second copy if one is already running.
  setsid sudo -u "$REMOTE_ACCOUNT" -H "$EDITOR_CACHE_HELPER" watch 300 \
    </dev/null >>"$PERSIST/editor-cache/watch.log" 2>&1 &
  disown 2>/dev/null || true
  ok "editor servers on container disk, extensions restored from the volume"
else
  warn "editor-cache missing, changed, or remote user absent -- refusing to execute it"
fi

# SSH: /root/.ssh stays local so a missing volume can never lock you out.
# Your own git key lives on the volume and is copied in with correct perms.
mkdir -p "$PERSIST/ssh"
chmod 700 "$PERSIST/ssh"
if [ -n "$(ls -A "$PERSIST/ssh" 2>/dev/null)" ]; then
  mkdir -p "$HOME_DIR/.ssh"
  chmod 700 "$HOME_DIR/.ssh"
  cp -a "$PERSIST/ssh/." "$HOME_DIR/.ssh/"
  chmod 600 "$HOME_DIR"/.ssh/id_* 2>/dev/null || true
  chmod 644 "$HOME_DIR"/.ssh/*.pub 2>/dev/null || true
  ok "git ssh keys restored from $PERSIST/ssh"
fi

# ------------------------------------------------------------ 2. shell env
log "Shell environment"
ok "persistent shell fragments reserved for the non-root remote user (not sourced as root)"

[ -f "$PERSIST/tmux.conf" ] && ln -sfn "$PERSIST/tmux.conf" "$HOME_DIR/.tmux.conf"
[ -f "$PERSIST/.gitconfig" ] && ln -sfn "$PERSIST/.gitconfig" "$HOME_DIR/.gitconfig"

# ----------------------------------------------------------- 3. apt basics
NEED=()
# btop lives in universe, nvtop in multiverse -- both are enabled on the stock
# RunPod pytorch image, so no add-apt-repository dance is needed here.
for p in tmux git git-lfs curl rsync htop btop nvtop jq nano sqlite3 shellcheck numactl zstd; do
  command -v "$p" >/dev/null 2>&1 || NEED+=("$p")
done
command -v rg >/dev/null 2>&1 || NEED+=(ripgrep)
# the bubblewrap package installs a binary called `bwrap`, not `bubblewrap`
command -v bwrap >/dev/null 2>&1 || NEED+=(bubblewrap)
# ncurses-term ships the several-hundred extra terminal descriptions that
# ncurses-base leaves out. It installs no binary, so it needs a dpkg test.
dpkg -s ncurses-term >/dev/null 2>&1 || NEED+=(ncurses-term)
if [ ${#NEED[@]} -gt 0 ]; then
  log "apt install: ${NEED[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq "${NEED[@]}" >/dev/null
fi
ok "base tools (tmux git git-lfs curl rsync htop btop nvtop jq nano rg sqlite3 shellcheck numactl zstd bwrap)"

# ----------------------------------------------- 3a. CUDA 13 compatibility
# morgott's locked encoder environment currently resolves PyTorch's CUDA 13
# wheels. Some RunPod images (including the 2x4090 image) ship a CUDA 12.8
# toolkit even though their R580 driver can run CUDA 13. Install the matching
# forward-compatibility userspace libraries so the experiment launcher gets
# the exact /usr/local/cuda-13.0/compat path it validates before resuming.
if [ -f "$MORGOTT_DIR/uv.lock" ] &&
  grep -q 'nvidia-cudnn-cu13' "$MORGOTT_DIR/uv.lock"; then
  if [ ! -d /usr/local/cuda-13.0/compat ]; then
    log "Installing CUDA 13 compatibility libraries"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null |
      head -1 | tr -d '[:space:]')
    compat_version=$(apt-cache madison cuda-compat-13-0 2>/dev/null |
      awk -v driver="$driver_version" '$3 ~ ("^" driver "-") { print $3; exit }')
    if [ -n "$compat_version" ]; then
      apt-get install -y -qq "cuda-compat-13-0=$compat_version" >/dev/null
    else
      warn "no CUDA 13 compatibility package exactly matches driver $driver_version; using repository candidate"
      apt-get install -y -qq cuda-compat-13-0 >/dev/null
    fi
  fi
  if [ -d /usr/local/cuda-13.0/compat ]; then
    ok "CUDA 13 compatibility libraries"
  else
    warn "CUDA 13 compatibility libraries missing -- morgott training will not resume"
  fi
fi

# ------------------------------------------------------- 3b. ghostty terminfo
# Ghostty sets TERM=xterm-ghostty, and Ubuntu 24.04 ships ncurses 6.4+20240113 --
# which predates ghostty's addition to the terminfo database, so ncurses-term does
# NOT supply it. Without an entry, every curses program dies before drawing:
#   clear -> "'xterm-ghostty': unknown terminal type"
#   nvtop -> "Error opening terminal: xterm-ghostty."
#   btop  -> exits silently with no output at all
# So synthesize a fallback aliased to xterm-256color. Deliberately minimal: only
# `Tc` (truecolor) is added on top, because a wrong escape string in a hand-written
# cap renders as garbage on screen, whereas a missing one just costs a feature.
# For an exact entry, run this from the ghostty machine instead -- it overwrites
# what we install here, and needs no root on the far end:
#   infocmp -x xterm-ghostty | ssh <pod> -- tic -x -
if ! infocmp xterm-ghostty >/dev/null 2>&1; then
  log "Adding xterm-ghostty terminfo fallback"
  tic -x -o /usr/share/terminfo - <<'EOT' 2>/dev/null || warn "tic failed -- ssh with TERM=xterm-256color"
xterm-ghostty|ghostty fallback (xterm-256color base + truecolor),
	Tc,
	use=xterm-256color,
EOT
fi
infocmp xterm-ghostty >/dev/null 2>&1 && ok "terminfo: xterm-ghostty resolves"

# git-lfs writes its filter config to ~/.gitconfig, which is symlinked to the
# volume and therefore survives -- but the hooks are per-clone and the binary is
# not, so re-run this. Without it, `git clone` silently yields 133-byte pointer
# stubs instead of weights and model loads fail with an opaque hash mismatch.
git lfs install --skip-repo >/dev/null 2>&1 && ok "git-lfs filters registered"

# ----------------------------------------------------------- 3c. github cli
# Ubuntu ships gh 2.45 (2024); the official repo tracks current. The binary is
# on the container disk and dies with it, but the auth token is not: gh writes
# to ~/.config/gh, and ~/.config is symlinked to $PERSIST/.config above. So a
# reset costs this re-install and nothing else -- no re-login.
if ! command -v gh >/dev/null 2>&1; then
  log "Installing GitHub CLI"
  KEYRING=/usr/share/keyrings/githubcli-archive-keyring.gpg
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg |
    dd of="$KEYRING" status=none
  chmod go+r "$KEYRING"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=$KEYRING] https://cli.github.com/packages stable main" \
    >/etc/apt/sources.list.d/github-cli.list
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gh >/dev/null
fi
if command -v gh >/dev/null 2>&1; then
  ok "gh $(gh --version 2>/dev/null | head -1 | awk '{print $3}')"
  if gh auth status >/dev/null 2>&1; then
    # Teaches git (and therefore git-lfs) to authenticate with the gh token.
    # Without it, `git lfs pull` on a private repo dies with
    # "could not read Username for 'https://github.com'".
    gh auth setup-git >/dev/null 2>&1 && ok "git credential helper -> gh"
  else
    warn "run 'gh auth login --web' to authenticate, then 'gh auth setup-git'"
  fi
fi

# ------------------------------------------------------------- 3d. azcopy
# morgott's data source of truth is Azure Blob, driven by scripts/azsync.sh --
# without azcopy on PATH every pull/push dies at "command not found", and a
# fresh pod has no corpus at all. Single static binary, so it goes to
# /usr/local/bin rather than a per-user ~/.local/bin: bootstrap runs as root but
# interactive and desktop sessions run as $REMOTE_USER, and both need it.
# Deliberately NOT azure-cli. Auth is the container SAS URL in the repo's .env,
# which sits on the volume and survives a reset; `az login` is only azsync's
# fallback for machines without a SAS, which this pod is not.
if ! command -v azcopy >/dev/null 2>&1; then
  log "Installing azcopy"
  AZ_URL=https://aka.ms/downloadazcopy-v10-linux
  [ "$(dpkg --print-architecture)" = arm64 ] && AZ_URL="$AZ_URL-arm64"
  AZ_TMP=$(mktemp -d)
  # the tarball wraps everything in a versioned dir (azcopy_linux_amd64_10.x.y/)
  if curl -fsSL "$AZ_URL" | tar xz -C "$AZ_TMP" --strip-components=1 2>/dev/null; then
    install -m 0755 "$AZ_TMP/azcopy" /usr/local/bin/azcopy
  fi
  rm -rf "$AZ_TMP"
fi
if command -v azcopy >/dev/null 2>&1; then
  ok "azcopy $(azcopy --version 2>/dev/null | awk '{print $3}')"
  ok "azcopy installed; morgott data auth is documented in data/README.md"
else
  warn "azcopy install failed -- scripts/azsync.sh cannot sync the corpus"
fi

# ------------------------------------------------------- 4. node (for codex)
if ! command -v node >/dev/null 2>&1 || [ "$(node -v | sed 's/v\([0-9]*\).*/\1/')" -lt 22 ]; then
  log "Installing Node.js 22"
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nodejs >/dev/null
fi
ok "node $(node -v)"

# ---------------------------------------------------------- 5. claude code
# Test for the REAL binary, not `command -v claude` -- $PERSIST/bin/claude is the
# --yolo wrapper and lives on the volume, so it survives a reset and would make
# `command -v` report claude as installed on a pod that has no Claude Code at all.
# The wrapper would then fail at runtime with "real binary not found on PATH".
if [ ! -x "$HOME_DIR/.local/bin/claude" ] && [ "${BOOTSTRAP_INSTALL_CLAUDE:-0}" = 1 ]; then
  log "Installing Claude Code"
  curl -fsSL https://claude.ai/install.sh | bash
fi
export PATH="$HOME_DIR/.local/bin:$PATH"
if [ -x "$HOME_DIR/.local/bin/claude" ]; then
  ok "claude $(claude --version 2>/dev/null || echo installed)"
  [ -f "$PERSIST/.claude/.credentials.json" ] || warn "run 'claude' once to log in"
elif [ "${BOOTSTRAP_INSTALL_CLAUDE:-0}" = 1 ]; then
  warn "Claude Code install failed"
else
  ok "Claude Code skipped (set BOOTSTRAP_INSTALL_CLAUDE=1 to opt in)"
fi

# --------------------------------------------------------------- 6. codex
CODEX_VERSION=${BOOTSTRAP_CODEX_VERSION:-0.147.0}
if ! command -v codex >/dev/null 2>&1; then
  log "Installing Codex CLI $CODEX_VERSION"
  npm install -g "@openai/codex@$CODEX_VERSION" --silent 2>/dev/null || warn "codex install failed"
fi
if command -v codex >/dev/null 2>&1; then
  ok "codex $(codex --version 2>/dev/null | tail -1 || echo installed)"
  [ -f "$PERSIST/.codex/auth.json" ] || warn "run 'codex login' to authenticate"
fi

# ------------------------------------------------------------------- 6a. uv
# uv ships in the stock RunPod pytorch image at /usr/bin/uv, so this is normally
# a no-op. It is installed here anyway because it is an *image* artifact we do
# not control, and section 6b is gated on `command -v uv`: on any image without
# it, every project venv would fail to restore and the only symptom would be
# 6b printing nothing at all. The explicit warn below is the point.
#
# INSTALLER_NO_MODIFY_PATH is required, not cosmetic -- the installer appends
# its own PATH line to .bashrc/.profile by default, which is exactly the fight
# section 7 exists to settle for the claude binary.
#
# /usr/local/bin, not the installer's default ~/.local/bin: bootstrap runs as
# root but venvs are synced as $REMOTE_USER, and that user needs the same uv.
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -fsSL https://astral.sh/uv/install.sh |
    env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null 2>&1 ||
    warn "uv install failed"
fi
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
else
  warn "uv missing -- project venvs below will NOT be restored"
fi

# ------------------------------------------------------ 6b. uv project venvs
# Python venvs are the worst possible shape for MooseFS: ~27k small files for a
# torch env, and the volume costs ~3.5 ms per file. Measured on morgott:
# `import sklearn, datasets, pandas` took 22 s from the volume, 6 s from the
# container disk. So venvs live on the container disk and each repo gets a
# `.venv` symlink pointing at it.
#
# The target directory MUST exist. uv creates the venv with mkdir, and mkdir on
# a DANGLING symlink returns EEXIST, not "no such file" -- the same trap that
# breaks codex with a dangling ~/.codex. Verified both ways:
#   .venv -> existing empty dir  =>  uv populates it            (works)
#   .venv -> missing dir         =>  "failed to create ... File exists" (fails)
# Empty is fine, missing is fatal, so mkdir the target every run.
#
# Bootstrap runs as root, but the desktop app and every interactive session run
# as $REMOTE_USER (see section 8). A root-owned venv is unusable by them --
# `uv sync` dies with EACCES -- so both the tree and the sync are handed to that
# user. UV_CACHE_DIR is on the volume, so a re-sync after a reset is seconds
# rather than a re-download; uv falls back from hardlink to copy across the
# volume/container-disk boundary, which is the intended trade here.
#
# Always run the locked sync, even when bin/python already exists. An empty or
# base-only venv still has a Python executable, which previously made bootstrap
# skip morgott's PyTorch/Transformers/Trackio stack on replacement pods.
# morgott is the active GPU project, so restore its exact experiment extras;
# other repositories keep their base-only environment.
VENV_ROOT=/opt/venvs
VENV_USER=$REMOTE_ACCOUNT
if command -v uv >/dev/null 2>&1 && [ -d "$VOL/code" ]; then
  log "Restoring uv project venvs on container disk"
  mkdir -p "$VENV_ROOT"
  id "$VENV_USER" >/dev/null 2>&1 && chown -R "$VENV_USER:$VENV_USER" "$VENV_ROOT"
  for proj in "$VOL"/code/*/; do
    [ -f "$proj/pyproject.toml" ] || continue
    name=$(basename "$proj")
    mkdir -p "$VENV_ROOT/$name"
    id "$VENV_USER" >/dev/null 2>&1 && chown "$VENV_USER:$VENV_USER" "$VENV_ROOT/$name"
    venv_link="${proj%/}/.venv"
    if [ -e "$venv_link" ] && [ ! -L "$venv_link" ]; then
      quarantine="$venv_link.bootstrap-migrated-$MIGRATION_STAMP"
      [ ! -e "$quarantine" ] || {
        echo "ERROR: migration quarantine already exists: $quarantine" >&2
        exit 1
      }
      mv "$venv_link" "$quarantine"
      warn "preserved replaced virtualenv at $quarantine"
    fi
    ln -sfnT "$VENV_ROOT/$name" "$venv_link"

    if [ "$name" = morgott ]; then
      sync_args="--locked --extra encoder --extra fa2 --extra tracking"
      log "  uv sync $sync_args ($name)"
    elif [ -x "$VENV_ROOT/$name/bin/python" ]; then
      ok "$name -> $VENV_ROOT/$name (already populated)"
      continue
    else
      sync_args="--locked"
      log "  uv sync $sync_args ($name)"
    fi
    if [ -f "$proj/uv.lock" ] && sudo -u "$VENV_USER" -H sh -c \
      "cd '$proj' && HF_HOME='$VOL/hf_cache' UV_CACHE_DIR='$VOL/.cache/uv' uv sync $sync_args" \
      >/dev/null 2>&1; then
      ok "$name -> $VENV_ROOT/$name (locked environment restored)"
      if [ "$name" = morgott ]; then
        if sudo -u "$VENV_USER" -H env HF_HOME="$VOL/hf_cache" \
          "$VENV_ROOT/$name/bin/python" -c \
          'import accelerate, kernels, peft, safetensors, torch, trackio, transformers; print(f"     torch={torch.__version__} transformers={transformers.__version__} trackio={trackio.__version__}")'; then
          ok "morgott encoder/tracking imports"
        else
          warn "morgott dependency import check failed"
        fi
      fi
    else
      warn "$name -- locked environment restore failed"
    fi
  done
fi

# ----------------------------------------- 7. root shell trust boundary (LAST)
# Persistent shell fragments are mutable user state. Keep them out of root's
# startup path; setup-remote-user.sh configures them for the non-root account.
log "Removing persistent fragments from root shell startup"
BRC="$HOME_DIR/.bashrc"
touch "$BRC"
sed -i '/^# >>> workspace env (bootstrap\.sh) >>>$/,/^# <<< workspace env (bootstrap\.sh) <<<$/d' "$BRC"
awk -v source_line="source $PERSIST/bashrc.sh" \
  -v env_line="[ -f $PERSIST/env.sh ] && . $PERSIST/env.sh" \
  '$0 != source_line && $0 != env_line { print }' "$BRC" >"$BRC.tmp"
mv "$BRC.tmp" "$BRC"
ok "root startup does not source persistent user shell code"

# ------------------------------------------- 8. non-root user for the desktop app
# Claude Code Desktop will not request Bypass Permissions when the remote account
# is uid 0 -- it silently downgrades the session to Accept Edits. The pod runs as
# root, so remote sessions come in as `ubuntu` instead. The account already
# exists because earlier restore steps need it; this completes its setup after
# the installers so it can copy root's claude binary into that user's ~/.local.
SETUP_REMOTE_USER_SHA256=67d42b8109ecfbacce095edd29ae2d5b808672595afb3b056571062577dfb401
SETUP_REMOTE_USER_HELPER="$MORGOTT_DIR/scripts/runpod-setup-remote-user.sh"
[ -x "$SETUP_REMOTE_USER_HELPER" ] || SETUP_REMOTE_USER_HELPER=/workspace/setup-remote-user.sh
if [ -x "$SETUP_REMOTE_USER_HELPER" ] &&
  printf '%s  %s\n' "$SETUP_REMOTE_USER_SHA256" "$SETUP_REMOTE_USER_HELPER" |
  sha256sum -c - >/dev/null 2>&1; then
  log "Setting up non-root user for Claude Code Desktop"
  RUSER="$REMOTE_ACCOUNT" bash "$SETUP_REMOTE_USER_HELPER" || warn "setup-remote-user.sh failed"
elif [ -e "$SETUP_REMOTE_USER_HELPER" ]; then
  warn "$SETUP_REMOTE_USER_HELPER changed -- refusing to execute it as root"
else
  warn "/workspace/setup-remote-user.sh missing -- desktop sessions will be root-only"
fi

# ----------------------------------------------------- 8a. agent skills and MCP
# This volume-owned helper runs without root authority. The canonical skills
# live in the shared persistent agent store established in section 1.
AGENT_SETUP="$VOL/agent-skills/setup.sh"
if [ -r "$AGENT_SETUP" ]; then
  log "Activating persistent agent skills and RunPod MCP servers"
  REMOTE_HOME=$(getent passwd "$REMOTE_ACCOUNT" | cut -d: -f6)
  sudo -u "$REMOTE_ACCOUNT" -H env \
    PATH="$REMOTE_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
    bash "$AGENT_SETUP" || warn "agent skill setup failed"
else
  warn "$AGENT_SETUP missing -- NVIDIA and RunPod skills are unavailable"
fi

# ------------------------------------------- 8b. morgott experiment readiness
# Restore services and validate the durable state, but never start or resume a
# multi-hour GPU workload from bootstrap. Training remains an explicit action.
MORGOTT_VENV="$VENV_ROOT/morgott"
MORGOTT_TRACKIO_DB="$VOL/hf_cache/trackio/morgott.db"
# Trackio 0.34 has no persisted project-level metric allowlist. The live
# `morgott` DB is a verified useful-metric projection; dated backups retain the
# complete raw history. These bounded URLs deliberately omit the dashboard
# write token.
MORGOTT_TRACKIO_TRAINING_VIEW='/?project=morgott&smoothing=0&sidebar=collapsed&hide_empty_tabs=true&metric_filter=%5E%28selection_rules%2F%7Cvalidation%2Fvalidation_morgott_%28positive%7Cnegative%29_source_label_macro_bce%7Ccheckpoint_diagnostics%2F%7Cval_bce_false_flags%2F%28banking77%7Charper_valley_bank%7Ctatqa%29%7Ctrain%2F%28loss%7Ctotal_loss%7Ccanonical_primary_loss%7Charmful_aux_loss%7Charmful_positive_loss%7Charmful_negative_loss%7Cpromptshield_loss%7Cpair_loss%7Cclip_fraction%7Cpre_clip_gradient_norm%7Cpeak_vram_gib%7Chead_lr%7Cadapter_lr%29%7Cperformance%2F%28optimizer_updates_per_second%7Cexamples_per_second%29%29%24'
MORGOTT_TRACKIO_DIAGNOSTICS_VIEW='/?project=morgott&smoothing=0&sidebar=collapsed&hide_empty_tabs=true&metric_filter=%5E%28val_bce_%28false_flags%7Cmissed_attacks%29%2F%7Ccheckpoint_diagnostics%2F%29'
MORGOTT_TRACKIO_EVALUATION_VIEW='/?project=morgott&smoothing=0&sidebar=collapsed&hide_empty_tabs=true&metric_filter=%5Eeval%2F%28canonical%7Cpromptshield%7Csep%7Csep_pairs%7Creal_finance%7Credteam%29%2F'
MORGOTT_TRACKIO_SUMMARY_VIEW='/?project=morgott&runs=summary-context-u17000-native-20260812%2Csummary-checkpoints-train1024-native-20260812&smoothing=0&sidebar=collapsed&hide_empty_tabs=true&metric_filter=%5Esummary%2F'
MORGOTT_TRACKIO_PLOT_ORDER='summary/canonical_tpr_at_1pct_fpr,summary/promptshield_tpr_at_1pct_fpr,summary/sep_tpr_at_1pct_fpr,summary/sep_pair_ordering,summary/finance_false_flags,summary/reserve_attested_recall,summary/reserve_bare_harmful_off_target_rate,summary/longcode_dev_clean_false_flags,selection_rules/ACTIVE_source_macro_blend,selection_rules/ACTIVE_micro_blend,selection_rules/ACTIVE_registered_blend,selection_rules/alt_source_macro_only,selection_rules/alt_worst_source,validation/validation_morgott_positive_source_label_macro_bce,validation/validation_morgott_negative_source_label_macro_bce,checkpoint_diagnostics/positive_recall_at_empirical_1pct_row_fpr,checkpoint_diagnostics/finance_false_positives_at_empirical_1pct_row_fpr,val_bce_false_flags/banking77,val_bce_false_flags/harper_valley_bank,val_bce_false_flags/tatqa,train/loss,train/canonical_primary_loss,train/promptshield_loss,train/pair_loss,train/clip_fraction,train/pre_clip_gradient_norm,train/peak_vram_gib,performance/optimizer_updates_per_second,performance/examples_per_second,gpu/mean_utilization,gpu/total_allocated_memory,gpu/total_power'
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  MORGOTT_TRACKIO_PUBLIC_BASE="https://${RUNPOD_POD_ID}-7860.proxy.runpod.net"
else
  MORGOTT_TRACKIO_PUBLIC_BASE='http://<pod-host>:7860'
fi
if [ -f "$MORGOTT_DIR/pyproject.toml" ]; then
  log "Validating morgott experiment environment"
  if [ -x "$MORGOTT_VENV/bin/python" ] &&
    env HF_HOME="$VOL/hf_cache" \
      MORGOTT_TRACKIO_DB="$MORGOTT_TRACKIO_DB" \
      "$MORGOTT_VENV/bin/python" - <<'PY'; then
import os
import sqlite3
from pathlib import Path

import accelerate
import kernels
import torch
import trackio
import transformers

print(
    "     morgott env: "
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"accelerate={accelerate.__version__} "
    f"trackio={getattr(trackio, '__version__', 'installed')}"
)
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access CUDA")
print(
    "     CUDA devices: "
    + ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
)

db_path = Path(os.environ["MORGOTT_TRACKIO_DB"])
if db_path.exists():
    connection = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=2)
    try:
        if connection.execute("PRAGMA quick_check(1)").fetchone()[0] != "ok":
            raise RuntimeError("Trackio SQLite quick_check failed")
        runs = connection.execute("SELECT COUNT(DISTINCT run_id) FROM configs").fetchone()[0]
        points = connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        print(f"     Trackio: quick_check=ok runs={runs} metric_rows={points}")
    finally:
        connection.close()
else:
    print("     Trackio: database absent")
PY
    ok "morgott CUDA and Trackio readiness"
  else
    warn "morgott environment validation failed -- do not launch GPU work"
  fi

  # The dashboard is a viewer over the existing SQLite project. Keep it on a
  # separate tmux socket so stopping/restarting it cannot kill a trainer. Keep
  # its write-token-bearing launch output in an ephemeral mode-0600 log.
  if [ "${MORGOTT_BOOTSTRAP_DASHBOARD:-1}" != 0 ] &&
    [ -x "$MORGOTT_VENV/bin/trackio" ] &&
    [ -f "$MORGOTT_TRACKIO_DB" ] &&
    command -v tmux >/dev/null 2>&1; then
    if id "$VENV_USER" >/dev/null 2>&1; then
      TRACKIO_AS=(sudo -u "$VENV_USER" -H)
    else
      TRACKIO_AS=()
    fi
    if "${TRACKIO_AS[@]}" tmux -L trackio has-session -t dash 2>/dev/null; then
      ok "Trackio dashboard already running on port 7860"
    else
      TRACKIO_LOG=/var/log/morgott-trackio.log
      install -o "$VENV_USER" -g "$VENV_USER" -m 0600 /dev/null "$TRACKIO_LOG"
      "${TRACKIO_AS[@]}" tmux -L trackio new-session -d -s dash \
        "cd '$MORGOTT_DIR' && exec env HF_HOME='$VOL/hf_cache' GRADIO_SERVER_PORT=7860 TRACKIO_PLOT_ORDER='$MORGOTT_TRACKIO_PLOT_ORDER' '$MORGOTT_VENV/bin/trackio' show --project morgott --host 0.0.0.0 --no-footer >'$TRACKIO_LOG' 2>&1"
      dashboard_ready=0
      for _attempt in 1 2 3 4 5 6 7 8 9 10; do
        if curl -fsS --max-time 2 http://127.0.0.1:7860/ >/dev/null 2>&1; then
          dashboard_ready=1
          break
        fi
        sleep 1
      done
      if "${TRACKIO_AS[@]}" tmux -L trackio has-session -t dash 2>/dev/null &&
        [ "$dashboard_ready" = 1 ]; then
        ok "Trackio dashboard started on port 7860"
      elif "${TRACKIO_AS[@]}" tmux -L trackio has-session -t dash 2>/dev/null; then
        warn "Trackio dashboard process is running but its HTTP health check failed"
      else
        warn "Trackio dashboard failed; private diagnostics are in $TRACKIO_LOG"
      fi
    fi
  fi
fi

# --------------------------------------------------------------- 9. report
echo
log "Ready."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/     GPU: /'
if [ -x "$MORGOTT_VENV/bin/python" ]; then
  "$MORGOTT_VENV/bin/python" -c 'import torch;print(f"     torch {torch.__version__} | cuda {torch.version.cuda} | avail {torch.cuda.is_available()}")' 2>/dev/null || true
else
  python -c 'import torch;print(f"     torch {torch.__version__} | cuda {torch.version.cuda} | avail {torch.cuda.is_available()}")' 2>/dev/null || true
fi
cat <<'EOT'

     projects   /workspace/projects       (persists)
     repos      /workspace/code           (persists)
     weights    /workspace/hf_cache       (persists, HF_HOME)
     home cfg   /workspace/home           (persists)

     t              tmux session "main" — always work inside it
     editor-cache   status of the Cursor/VS Code extension snapshot
     claude --yolo  no permission prompts  (alias: yolo)
     codex  --yolo  no approval prompts    (alias: cyolo)
     gpu            live nvidia-smi
     btop           cpu/mem/disk/net monitor (has a gpu-totals panel)
     nvtop          per-GPU utilisation, memory and process list

     morgott dashboard  use one of the curated URLs below
     morgott GPU work   explicit only; bootstrap never launches training or evaluation

EOT

printf '     morgott training view  %s%s\n' "$MORGOTT_TRACKIO_PUBLIC_BASE" "$MORGOTT_TRACKIO_TRAINING_VIEW"
printf '     morgott diagnostics    %s%s\n' "$MORGOTT_TRACKIO_PUBLIC_BASE" "$MORGOTT_TRACKIO_DIAGNOSTICS_VIEW"
printf '     morgott eval view      %s%s\n' "$MORGOTT_TRACKIO_PUBLIC_BASE" "$MORGOTT_TRACKIO_EVALUATION_VIEW"
printf '     morgott summary view   %s%s\n' "$MORGOTT_TRACKIO_PUBLIC_BASE" "$MORGOTT_TRACKIO_SUMMARY_VIEW"
