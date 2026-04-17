# Project Instructions

## Auto commit & push

Every time a coding task finishes (implementation complete, no more pending changes), automatically:
1. `git add` the changed files
2. `git commit` with a descriptive message
3. `./push.sh`

Do not ask for confirmation — just do it after each coding round.

## Python module resolution

The repo directory (`gr_demo/`) **is** the Python package — it has `__init__.py` at the root.
To import `gr_demo.*` from standalone scripts under `experiments/scripts/`, you must add the
**parent of the repo root** to `sys.path`, NOT the repo root itself:

```python
# In experiments/scripts/some_script.py:
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(repo_root))  # adds parent of gr_demo/
```

The CLI entry point is always `python run.py <command>`, NOT `python -m gr_demo`.
