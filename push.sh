#!/bin/bash
# ==============================================================================
# One-click push script (unified identity — no more author rewriting)
# ==============================================================================
#
# Usage:
#   ./push.sh                          # push all (main + sensitive)
#   ./push.sh -m "commit message"      # commit + push all
#   ./push.sh --config-only         # only push config/
#   ./push.sh --main-only              # only push main repo
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENSITIVE_DIR="${SCRIPT_DIR}/sensitive"

COMMIT_MSG=""
SENSITIVE_ONLY=false
MAIN_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--message)    COMMIT_MSG="$2"; shift 2 ;;
        --config-only) SENSITIVE_ONLY=true; shift ;;
        --main-only)      MAIN_ONLY=true; shift ;;
        *)
            echo "Usage: ./push.sh [-m 'message'] [--config-only] [--main-only]"
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

push_main() {
    cd "$SCRIPT_DIR"
    push_repo "main repo"
}

push_config() {
    if [ ! -d "${SENSITIVE_DIR}/.git" ]; then
        echo "Error: config/.git not found. Run 'cd sensitive && git init' first."
        return 1
    fi
    cd "$SENSITIVE_DIR"
    push_repo "config/"
    cd "$SCRIPT_DIR"
}

echo "============================================================"
echo "Push Script   $(date '+%Y-%m-%d %H:%M:%S')"
if [ -n "$COMMIT_MSG" ]; then echo "Commit: $COMMIT_MSG"; fi
echo "============================================================"

if [ "$SENSITIVE_ONLY" = true ]; then
    push_config
elif [ "$MAIN_ONLY" = true ]; then
    push_main
else
    push_main
    push_config
fi

echo ""
echo "============================================================"
echo "All done!"
echo "============================================================"
