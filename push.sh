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

# Load private config (company git identity, etc.)
source "${SENSITIVE_DIR}/config.sh"

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

push_to_company_remote() {
    local BRANCH="$1"
    local REMOTE="$2"
    local MIRROR_NAME="$3"
    local MIRROR_EMAIL="$4"

    if ! git remote | grep -q "^${REMOTE}$"; then
        echo "  Skipping ${REMOTE} (remote not configured)"
        return 0
    fi

    echo "  Mirroring to ${REMOTE} (rewriting author as ${MIRROR_NAME})..."
    local TEMP_BRANCH="_mirror_${REMOTE}_$$"

    # Step 1: Fetch remote and incorporate GPU machine's commits
    git fetch "$REMOTE" "$BRANCH" --quiet 2>/dev/null || true
    local REMOTE_HEAD
    REMOTE_HEAD="$(git rev-parse "${REMOTE}/${BRANCH}" 2>/dev/null || echo "")"

    if [ -n "$REMOTE_HEAD" ]; then
        if ! git merge-base --is-ancestor "$REMOTE_HEAD" HEAD 2>/dev/null; then
            echo "  Incorporating GPU commits from ${REMOTE}..."
            git rebase --quiet "${REMOTE}/${BRANCH}" 2>/dev/null || {
                echo "  Warning: rebase onto ${REMOTE} failed, trying merge..."
                git rebase --abort --quiet 2>/dev/null || true
                git merge "${REMOTE}/${BRANCH}" --no-edit --quiet 2>/dev/null || {
                    echo "  Warning: merge failed too, pushing as-is..."
                    git merge --abort 2>/dev/null || true
                }
            }
        fi
    fi

    # Step 2: Find merge-base for rewrite range
    local BASE
    BASE="$(git merge-base "${REMOTE}/${BRANCH}" HEAD 2>/dev/null || echo "")"

    # Robust cleanup
    cleanup_mirror() {
        git rebase --abort --quiet 2>/dev/null || true
        git checkout -f "$BRANCH" --quiet 2>/dev/null || true
        git branch -D "$TEMP_BRANCH" --quiet 2>/dev/null || true
        if ! git symbolic-ref HEAD >/dev/null 2>&1; then
            git checkout -f master --quiet 2>/dev/null || true
        fi
        if [ "$STASHED" = true ]; then
            if ! git stash pop --quiet 2>/dev/null; then
                echo "  ERROR: stash pop failed. Your changes are in 'git stash list'."
            fi
            STASHED=false
        fi
    }
    trap 'cleanup_mirror; trap - INT TERM; return 1' INT TERM

    # Stash uncommitted changes
    local STASHED=false
    if ! git diff --quiet || ! git diff --cached --quiet; then
        git stash push --quiet -m "push.sh: auto-stash before ${REMOTE} mirror"
        STASHED=true
    fi

    git checkout -b "$TEMP_BRANCH" "$BRANCH" --quiet

    # Step 3: Rewrite only non-matching-author commits
    local REWRITE_EXEC="if [ \"\$(git log -1 --format='%ae')\" != '$MIRROR_EMAIL' ]; then GIT_COMMITTER_NAME='$MIRROR_NAME' GIT_COMMITTER_EMAIL='$MIRROR_EMAIL' git commit --amend --no-edit --quiet --author='$MIRROR_NAME <$MIRROR_EMAIL>'; fi"

    local REBASE_OK=true
    if [ -n "$BASE" ]; then
        GIT_SEQUENCE_EDITOR=true git rebase --quiet --onto "$BASE" "$BASE" "$TEMP_BRANCH" \
            --exec "$REWRITE_EXEC" \
            2>/dev/null || REBASE_OK=false
    else
        GIT_SEQUENCE_EDITOR=true git rebase --quiet --root "$TEMP_BRANCH" \
            --exec "$REWRITE_EXEC" \
            2>/dev/null || REBASE_OK=false
    fi

    if [ "$REBASE_OK" = false ]; then
        echo "  Warning: rebase failed, aborting..."
        git rebase --abort --quiet 2>/dev/null || true
        echo "  Falling back: pushing original commits to ${REMOTE}..."
        git push "$REMOTE" "$BRANCH:$BRANCH" --force 2>&1 || echo "  Warning: fallback push to ${REMOTE} failed"
    else
        git push "$REMOTE" "$TEMP_BRANCH:$BRANCH" --force 2>&1 || echo "  Warning: push to ${REMOTE} failed"
    fi

    # Cleanup
    trap - INT TERM
    cleanup_mirror

    local FINAL_BRANCH
    FINAL_BRANCH="$(git branch --show-current 2>/dev/null || echo "")"
    if [ "$FINAL_BRANCH" != "$BRANCH" ]; then
        echo "  ERROR: expected '$BRANCH' but on '${FINAL_BRANCH:-DETACHED HEAD}'. Recovering..."
        git checkout -f "$BRANCH" --quiet 2>/dev/null || git checkout -f master --quiet
    fi
    echo "  ${REMOTE} mirror complete."
}

push_main() {
    echo ""
    echo "============================================================"
    echo "Pushing main repo..."
    echo "============================================================"

    cd "$SCRIPT_DIR"

    # Safety: recover from detached HEAD (e.g. interrupted rebase from previous run)
    if ! git symbolic-ref HEAD >/dev/null 2>&1; then
        echo "  WARNING: detached HEAD detected! Likely from a previous interrupted push."
        echo "  Recovering: switching back to master..."
        git rebase --abort --quiet 2>/dev/null || true
        git checkout master --quiet
    fi

    local BRANCH
    BRANCH="$(git branch --show-current)"

    # Safety: if on a stale _mirror_company_ branch, switch back to master
    if [[ "$BRANCH" == _mirror_* ]]; then
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
    if git remote | grep -q '^personal$'; then
        echo "  Pulling from personal (rebase)..."
        git pull --rebase personal "$BRANCH" 2>&1 || echo "  Warning: pull from personal failed"
        echo "  Pushing to personal..."
        git push personal "$BRANCH" 2>&1 || echo "  Warning: push to personal failed"
    else
        echo "  Skipping personal (remote not configured)"
    fi

    # Push to public remotes (rewrite author for local commits, preserve GPU commits)
    push_to_company_remote "$BRANCH" "company" \
        "${GIT_AUTHOR_NAME:?Missing GIT_AUTHOR_NAME in config/config.sh}" \
        "${GIT_AUTHOR_EMAIL:?Missing GIT_AUTHOR_EMAIL in config/config.sh}"

    push_to_company_remote "$BRANCH" "origin" \
        "${COMPANY2_GIT_NAME:?Missing COMPANY2_GIT_NAME in config/config.sh}" \
        "${COMPANY2_GIT_EMAIL:?Missing COMPANY2_GIT_EMAIL in config/config.sh}"

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
    local COMPANY_NAME="${GIT_AUTHOR_NAME:?Missing GIT_AUTHOR_NAME in config/config.sh}"
    local COMPANY_EMAIL="${GIT_AUTHOR_EMAIL:?Missing GIT_AUTHOR_EMAIL in config/config.sh}"

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

    # Pull --rebase then push to all remotes
    local CURRENT_BRANCH
    CURRENT_BRANCH="$(git branch --show-current)"
    for remote in $(git remote); do
        echo "  Pulling from ${remote} (rebase)..."
        git pull --rebase "$remote" "$CURRENT_BRANCH" 2>&1 || echo "  Warning: pull from ${remote} failed"
        echo "  Pushing to ${remote}..."
        git push "$remote" "$CURRENT_BRANCH" 2>&1 || echo "  Warning: push to ${remote} failed"
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
