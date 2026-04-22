"""Convenience entry point — run from inside the repo root.

Usage:
    python run.py eval-all --only-sid
    python run.py train --model qwen3-0.6b
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import main

if __name__ == '__main__':
    main()
