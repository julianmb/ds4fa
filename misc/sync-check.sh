#!/bin/sh
# sync-check.sh: enforce the "stay close to upstream" fork policy.
#
# Fetches antirez/ds4 and reports how far this fork has diverged. It fails if:
#   - upstream cannot be fetched, or
#   - this fork is behind upstream by more than FORK_BEHIND_MAX commits, or
#   - this fork is ahead by more than FORK_AHEAD_MAX commits AND FORK_NOTES.md
#     has not been modified more recently than the divergence base.
#
# Intended for CI. Override thresholds via env vars.

set -e

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_REF="${UPSTREAM_REF:-main}"
FORK_BEHIND_MAX="${FORK_BEHIND_MAX:-50}"
FORK_AHEAD_MAX="${FORK_AHEAD_MAX:-200}"

cd "$(dirname "$0")/.."

if ! git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
    echo "sync-check: no '$UPSTREAM_REMOTE' remote configured; skipping."
    exit 0
fi

git fetch -q "$UPSTREAM_REMOTE" "$UPSTREAM_REF"

BASE="$(git merge-base HEAD "$UPSTREAM_REMOTE/$UPSTREAM_REF")"
BEHIND="$(git rev-list --count "$BASE".."$UPSTREAM_REMOTE/$UPSTREAM_REF")"
AHEAD="$(git rev-list --count "$BASE"..HEAD)"

echo "sync-check: behind upstream by $BEHIND commit(s), ahead by $AHEAD commit(s)."

if [ "$BEHIND" -gt "$FORK_BEHIND_MAX" ]; then
    echo "sync-check: FAIL: $BEHIND commits behind upstream (max $FORK_BEHIND_MAX)."
    echo "sync-check: merge or rebase antirez/ds4 and update FORK_NOTES.md."
    exit 1
fi

if [ "$AHEAD" -gt "$FORK_AHEAD_MAX" ]; then
    NOTES="$(git log --format=%H -1 -- FORK_NOTES.md || true)"
    if [ -z "$NOTES" ] || [ "$(git merge-base "$NOTES" "$BASE")" != "$BASE" ] && \
       [ "$(git rev-list --count "$BASE".."$NOTES")" -eq 0 ]; then
        echo "sync-check: FAIL: $AHEAD commits ahead of upstream (max $FORK_AHEAD_MAX)"
        echo "sync-check: without a recent FORK_NOTES.md update. Document the"
        echo "sync-check: divergence before merging."
        exit 1
    fi
    echo "sync-check: ahead by $AHEAD (over threshold) but FORK_NOTES.md is current."
fi

echo "sync-check: OK"
