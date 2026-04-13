#!/bin/bash
# ==============================================================================
# One-click push script for dual-remote setup
# ==============================================================================
#
# Usage:
#   ./push.sh                          # push all (main + sensitive)
#   ./push.sh -m "commit message"      # commit + push all
#   ./push.sh --config-only          # only push config/
#   ./push.sh --main-only              # only push main repo
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENSITIVE_DIR="${SCRIPT_DIR}/sensitive"

COMMIT_MSG=""
SENSITIVE_ONLY=false
MAIN_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--message)
            COMMIT_MSG="$2"
            shift 2
            ;;
        --config-only)
            SENSITIVE_ONLY=true
            shift
            ;;
        --main-only)
            MAIN_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./push.sh [-m 'message'] [--config-only] [--main-only]"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Helper functions
# ==============================================================================

push_main() {
    echo ""
    echo "============================================================"
    echo "Pushing main repo..."
    echo "============================================================"

    cd "$SCRIPT_DIR"

    local BRANCH
    BRANCH="$(git branch --show-current)"

    # Safety: if on a stale _mirror_company_ branch, switch back to master
    if [[ "$BRANCH" == _mirror_company_* ]]; then
        echo "  Warning: stuck on temp branch $BRANCH, switching to master..."
        git checkout master --quiet
        git branch -D "$BRANCH" --quiet 2>/dev/null || true
        BRANCH="master"
    fi

    # Commit if message provided
    if [ -n "$COMMIT_MSG" ]; then
        git add -A
        if ! git diff --cached --quiet; then
            git commit -m "$COMMIT_MSG"
            echo "Committed: $COMMIT_MSG"
        else
            echo "Nothing to commit in main repo"
        fi
    fi

    # Push to personal (原始 commit，个人身份)
    echo "  Pushing to personal..."
    git push personal "$BRANCH" 2>&1 || echo "  Warning: push to personal failed"

    # Push to company (重写 author 为公司身份)
    echo "  Mirroring to company (rewriting author)..."
    local COMPANY_NAME="Company User"
    local COMPANY_EMAIL="user@company.com"
    local TEMP_BRANCH="_mirror_company_$$"

    # 找到 company 和 personal 的分叉点，只重写之后的 commits
    local BASE
    BASE="$(git merge-base "company/${BRANCH}" "$BRANCH" 2>/dev/null || echo "")"

    # Ensure cleanup on interrupt
    trap 'git checkout "$BRANCH" --quiet 2>/dev/null; git branch -D "$TEMP_BRANCH" --quiet 2>/dev/null; trap - INT TERM; return 1' INT TERM

    git checkout -b "$TEMP_BRANCH" "$BRANCH" --quiet

    if [ -n "$BASE" ]; then
        # 重写 base 之后的 commits
        GIT_SEQUENCE_EDITOR=true git rebase --quiet --onto "$BASE" "$BASE" "$TEMP_BRANCH" \
            --exec "GIT_COMMITTER_NAME='$COMPANY_NAME' GIT_COMMITTER_EMAIL='$COMPANY_EMAIL' git commit --amend --no-edit --quiet --author='$COMPANY_NAME <$COMPANY_EMAIL>'" \
            2>/dev/null || true
    else
        # 没有共同祖先，重写所有 commits
        GIT_SEQUENCE_EDITOR=true git rebase --quiet --root "$TEMP_BRANCH" \
            --exec "GIT_COMMITTER_NAME='$COMPANY_NAME' GIT_COMMITTER_EMAIL='$COMPANY_EMAIL' git commit --amend --no-edit --quiet --author='$COMPANY_NAME <$COMPANY_EMAIL>'" \
            2>/dev/null || true
    fi

    git push company "$TEMP_BRANCH:$BRANCH" --force 2>&1 || echo "  Warning: push to company failed"

    # Cleanup
    trap - INT TERM
    git checkout "$BRANCH" --quiet
    git branch -D "$TEMP_BRANCH" --quiet
    echo "  Company mirror complete."

    echo "Main repo push complete."
}

push_config() {
    echo ""
    echo "============================================================"
    echo "Pushing config/ repo..."
    echo "============================================================"

    cd "$SENSITIVE_DIR"

    # Check if git is initialized
    if [ ! -d ".git" ]; then
        echo "Error: config/.git not found. Run 'cd sensitive && git init' first."
        return 1
    fi

    # Company identity for private config
    local COMPANY_NAME="Company User"
    local COMPANY_EMAIL="user@company.com"

    # Commit if message provided (使用公司身份)
    if [ -n "$COMMIT_MSG" ]; then
        git add -A
        if ! git diff --cached --quiet; then
            git -c user.name="$COMPANY_NAME" -c user.email="$COMPANY_EMAIL" commit -m "$COMMIT_MSG"
            echo "Committed (as $COMPANY_NAME): $COMMIT_MSG"
        else
            echo "Nothing to commit in config/"
        fi
    fi

    # Push to all remotes
    for remote in $(git remote); do
        echo "  Pushing to ${remote}..."
        git push "$remote" "$(git branch --show-current)" 2>&1 || echo "  Warning: push to ${remote} failed"
    done

    echo "Sensitive repo push complete."
    cd "$SCRIPT_DIR"
}

# ==============================================================================
# Execute
# ==============================================================================

echo "============================================================"
echo "Push Script"
echo "============================================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
if [ -n "$COMMIT_MSG" ]; then
    echo "Commit message: $COMMIT_MSG"
fi
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
