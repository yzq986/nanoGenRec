"""Convenience entry point — run from inside the repo root.

Usage:
    python run.py eval-all --only-sid
    python run.py train --model qwen3-0.6b
"""

import sys
import os

# 把上级目录加到 Python path，这样 gr_demo 包可被发现
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gr_demo.cli import main

if __name__ == '__main__':
    main()
