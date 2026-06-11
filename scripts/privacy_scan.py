#!/usr/bin/env python3
"""Repository privacy/secret scan for release hygiene.

The scanner is intentionally conservative: high-severity findings fail the
command, while known public references and documented local environment paths
are reported as low-severity notes.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "public_benchmarks/data",
    "public_benchmarks/runs",
    "experiments/sid_cache",
    "experiments/ntp_data",
    "experiments/ntp_checkpoints",
    "papers",
}

TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
    "",
}


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    severity: str


RULES = [
    Rule("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "high"),
    Rule("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "high"),
    Rule("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "high"),
    Rule("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "high"),
    Rule("openai-style-token", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"), "high"),
    Rule("credential-assignment", re.compile(r"(?i)\b(password|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"), "high"),
    Rule("internal-company-host", re.compile(r"(?i)(company-gitlab|alibabacloudcs\.com|devops\.[A-Za-z0-9.-]+)"), "high"),
    Rule("absolute-user-path", re.compile(r"(/Users/[^ \n\t`'\"]+|/home/dev/[^ \n\t`'\"]+|/root/[^ \n\t`'\"]+)"), "low"),
    Rule("author-email", re.compile(r"\byeziqing986@gmail\.com\b"), "low"),
]


ALLOW_LOW_NAMES = {"absolute-user-path", "author-email"}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def should_skip(path: Path) -> bool:
    if path.resolve() == Path(__file__).resolve():
        return True
    relative = rel(path)
    parts = set(Path(relative).parts)
    if parts & {".git", ".pytest_cache", "__pycache__"}:
        return True
    return any(relative == d or relative.startswith(f"{d}/") for d in SKIP_DIRS)


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_dir() or should_skip(path):
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        files.append(path)
    return sorted(files)


def scan_file(path: Path) -> list[tuple[str, str, int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    findings = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for rule in RULES:
            if rule.pattern.search(line):
                severity = "low" if rule.name in ALLOW_LOW_NAMES else rule.severity
                findings.append((severity, rule.name, lineno, line.strip()[:180]))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-low", action="store_true", help="Print low-severity allowlisted findings.")
    args = parser.parse_args()

    high: list[tuple[str, str, int, str]] = []
    low: list[tuple[str, str, int, str]] = []
    for path in iter_files():
        for severity, name, lineno, line in scan_file(path):
            item = (rel(path), name, lineno, line)
            if severity == "high":
                high.append(item)
            else:
                low.append(item)

    if high:
        print("High-severity privacy findings:")
        for path, name, lineno, line in high:
            print(f"  {path}:{lineno} [{name}] {line}")
    else:
        print("No high-severity privacy findings.")

    if args.show_low and low:
        print("\nLow-severity reviewed references:")
        for path, name, lineno, line in low:
            print(f"  {path}:{lineno} [{name}] {line}")

    print(f"\nScanned {len(iter_files())} text files. high={len(high)} low={len(low)}")
    return 1 if high else 0


if __name__ == "__main__":
    sys.exit(main())
