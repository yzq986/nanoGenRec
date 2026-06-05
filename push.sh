#!/bin/bash
# ==============================================================================
# One-click push script (unified identity — no more author rewriting)
# ==============================================================================
#
# Usage:
#   ./push.sh                          # push current branch to all configured remotes
#   ./push.sh -m "commit message"      # commit + push current branch
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--message)    COMMIT_MSG="$2"; shift 2 ;;
        *)
            echo "Usage: ./push.sh [-m 'message']"
            exit 1
            ;;
    esac
done

push_repo() {
    local label="$1"
    echo ""
    echo "============================================================"
    echo "Pushing ${label}..."
    echo "============================================================"

    local BRANCH
    BRANCH="$(git branch --show-current)"
    if [ -z "$BRANCH" ]; then
        echo "  ERROR: detached HEAD, aborting."
        return 1
    fi

    if [ -n "$COMMIT_MSG" ]; then
        git add -A
        if ! git diff --cached --quiet; then
            git commit -m "$COMMIT_MSG"
            echo "Committed: $COMMIT_MSG"
        else
            echo "Nothing to commit."
        fi
    fi

    for remote in $(git remote); do
        echo "  [${remote}] pull --rebase..."
        git pull --rebase "$remote" "$BRANCH" 2>&1 | tail -3 || echo "  Warning: pull from ${remote} failed"
        echo "  [${remote}] push..."
        git push "$remote" "$BRANCH" 2>&1 | tail -3 || echo "  Warning: push to ${remote} failed"
    done

    echo "${label} push complete."
}

echo "============================================================"
echo "Push Script   $(date '+%Y-%m-%d %H:%M:%S')"
if [ -n "$COMMIT_MSG" ]; then echo "Commit: $COMMIT_MSG"; fi
echo "============================================================"

cd "$SCRIPT_DIR"
push_repo "repository"

echo ""
echo "============================================================"
echo "All done!"
echo "============================================================"
