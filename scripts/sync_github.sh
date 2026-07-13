#!/bin/bash
# sync_github.sh — keep rossby_palooza in sync with GitHub (SSH).
#
# Called by the Claude Code SessionEnd hook (settings.json) and runnable
# manually anytime:  bash scripts/sync_github.sh
#
# Behavior:
#   1. flock: only one sync at a time (concurrent session ends are no-ops).
#   2. Fast exit when there is nothing to do (clean tree + up-to-date).
#   3. git pull --rebase --autostash; on ANY conflict: abort the rebase,
#      log it, exit non-zero WITHOUT pushing — a human resolves. Never
#      force-pushes.
#   4. git add -A; leak/size guard (token/credential/key patterns, files
#      >40MB); commit only if staged changes exist; push.
#
# Log: logs/sync.log (gitignored).
set -uo pipefail

REPO="/net/flood/data2/users/x_yan/rossby_palooza"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/sync.log"
LOCK="$LOG_DIR/sync.lock"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

exec 9>"$LOCK"
if ! flock -n 9; then
    # Another sync is running; that one will pick up our changes.
    exit 0
fi

cd "$REPO" || { log "ERROR: cannot cd to $REPO"; exit 1; }

# --- fast path: nothing local to commit and nothing remote to pull ---------
git fetch -q origin main 2>>"$LOG" || { log "ERROR: fetch failed (network?)"; exit 1; }
DIRTY=$(git status --porcelain | head -1)
BEHIND=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
AHEAD=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
if [ -z "$DIRTY" ] && [ "$BEHIND" -eq 0 ] && [ "$AHEAD" -eq 0 ]; then
    exit 0
fi

# --- integrate remote first (rebase; abort on conflict, never force) -------
if [ "$BEHIND" -gt 0 ]; then
    if ! git pull --rebase --autostash -q origin main >> "$LOG" 2>&1; then
        git rebase --abort >> "$LOG" 2>&1 || true
        log "CONFLICT: pull --rebase failed; aborted, NOT pushing. Resolve manually."
        exit 2
    fi
fi

# --- stage + guards ---------------------------------------------------------
git add -A

LEAK=$(git diff --cached --name-only | grep -iE '_token|\.pem$|\.key$|credential|cdsapirc|\.netrc|id_ed25519|id_rsa' | head -5)
if [ -n "$LEAK" ]; then
    log "BLOCKED: suspicious filenames staged, NOT committing: $LEAK"
    git reset -q
    exit 3
fi
BIG=$(git diff --cached --name-only -z | xargs -0 -I{} find "{}" -maxdepth 0 -type f -size +40M 2>/dev/null | head -3)
if [ -n "$BIG" ]; then
    log "BLOCKED: files >40MB staged, NOT committing: $BIG"
    git reset -q
    exit 3
fi

# --- commit (only if there is something) + push -----------------------------
if ! git diff --cached --quiet; then
    git commit -q -m "auto-sync: $(date '+%Y-%m-%d %H:%M') @$(hostname -s)" >> "$LOG" 2>&1
fi
if [ "$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)" -gt 0 ]; then
    if git push -q origin main >> "$LOG" 2>&1; then
        log "pushed $(git rev-parse --short HEAD)"
    else
        log "ERROR: push failed (see above)."
        exit 1
    fi
fi
exit 0
