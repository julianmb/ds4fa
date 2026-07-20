#!/bin/sh
# sync.sh: rebase this Strix Halo fork onto the latest upstream DS4.
#
# Fetches antirez/ds4, runs the divergence policy check, then rebases the
# current branch onto upstream/main. Safe by default: it aborts if the working
# tree is dirty or if sync-check.sh would fail.
#
# Usage: misc/sync.sh [branch]   (default branch: main)

set -e

BRANCH="${1:-main}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_REF="${UPSTREAM_REF:-main}"

cd "$(dirname "$0")/.."

if ! git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
    echo "sync: adding upstream remote $UPSTREAM_REMOTE -> antirez/ds4"
    git remote add "$UPSTREAM_REMOTE" https://github.com/antirez/ds4.git
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "sync: working tree is dirty; commit or stash changes first." >&2
    exit 1
fi

echo "sync: fetching upstream..."
git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_REF"

if [ -f misc/sync-check.sh ]; then
    echo "sync: running divergence policy check (pre-rebase)..."
    sh misc/sync-check.sh || {
        echo "sync: sync-check.sh failed; refusing to rebase." >&2
        exit 1
    }
fi

echo "sync: rebasing $BRANCH onto $UPSTREAM_REMOTE/$UPSTREAM_REF..."
git rebase "$UPSTREAM_REMOTE/$UPSTREAM_REF" "$BRANCH"

echo "sync: done. Review the result, then 'git push --force-with-lease origin $BRANCH'."
