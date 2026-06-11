#!/usr/bin/env python3
"""Lightweight consistency checks for the nanoGenRec paper draft.

The checks intentionally focus on facts that are easy to validate from the
repository: author placeholders, citation keys, figure paths, and table values
copied from experiment logs. They are not a substitute for human review.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = PAPER / "nanogenrec.tex"
BIB = PAPER / "references.bib"


def normalize(text: str) -> str:
    replacements = {
        r"\%": "%",
        r"\sim": "~",
        r"\textsc{nanoGenRec}": "nanoGenRec",
        r"\texttt{yeziqing986@gmail.com}": "yeziqing986@gmail.com",
        "$": "",
        "**": "",
        "`": "",
        "{": "",
        "}": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def line_with(text: str, token: str, values: list[str] | None = None) -> str | None:
    token_norm = normalize(token)
    values_norm = [normalize(value) for value in values or []]
    for line in text.splitlines():
        line_norm = normalize(line)
        if token_norm in line_norm and all(value in line_norm for value in values_norm):
            return line
    return None


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def check_contains(text: str, token: str, values: list[str], where: str, failures: list[str]) -> None:
    line = line_with(text, token, values)
    if line is None:
        closest = line_with(text, token)
        suffix = f"; closest={closest.strip()!r}" if closest else ""
        fail(f"{where}: missing row/token {token!r} with values {values!r}{suffix}", failures)


@dataclass(frozen=True)
class NumericCheck:
    name: str
    source_path: str
    source_token: str
    tex_token: str
    values: tuple[str, ...]


NUMERIC_CHECKS = [
    # EXP-015 model scaling table.
    NumericCheck("EXP-015 scale-01", "experiments/logs/exp-015.md", "scale-01", "scale-01", ("1.7M", "235.1", "5.460", "1.9%", "23.6%")),
    NumericCheck("EXP-015 scale-02", "experiments/logs/exp-015.md", "scale-02", "scale-02", ("3.6M", "100.4", "4.609", "3.7%", "31.7%")),
    NumericCheck("EXP-015 scale-03", "experiments/logs/exp-015.md", "scale-03", "scale-03", ("5.1M", "69.6", "4.243", "5.4%", "45.6%")),
    NumericCheck("EXP-015 scale-04", "experiments/logs/exp-015.md", "scale-04", "scale-04", ("17.5M", "28.1", "3.334", "9.8%", "60.5%")),
    NumericCheck("EXP-015 scale-05", "experiments/logs/exp-015.md", "scale-05", "scale-05", ("34.5M", "24.0", "3.178", "11.5%", "62.5%")),
    NumericCheck("EXP-015 scale-06", "experiments/logs/exp-015.md", "scale-06", "scale-06", ("71.6M", "20.8", "3.037", "12.6%", "66.2%")),
    NumericCheck("EXP-015 scale-07", "experiments/logs/exp-015.md", "scale-07", "scale-07", ("101.1M", "19.4", "2.965", "13.7%", "65.8%")),
    NumericCheck("EXP-015 scaling law", "experiments/logs/exp-015.md", "L̂(N) = 2.522", "L(N)", ("2.522", "2055.1", "0.456")),
    # EXP-016 behavior-window scaling table.
    NumericCheck("EXP-016 A-7d-S", "experiments/logs/exp-016.md", "A-7d-S", "A-7d-S", ("7", "65M", "1.02M", "30.60", "62.1%")),
    NumericCheck("EXP-016 B-14d-S", "experiments/logs/exp-016.md", "B-14d-S", "B-14d-S", ("14", "130M", "1.69M", "27.05", "58.5%")),
    NumericCheck("EXP-016 C-31d-S", "experiments/logs/exp-016.md", "C-31d-S", "C-31d-S", ("31", "262M", "3.04M", "28.05", "60.5%")),
    NumericCheck("EXP-016 D-62d-S", "experiments/logs/exp-016.md", "D-62d-S", "D-62d-S", ("62", "441M", "4.86M", "30.03", "58.6%")),
    NumericCheck("EXP-016 E-90d-S", "experiments/logs/exp-016.md", "E-90d-S", "E-90d-S", ("90", "553M", "6.18M", "31.89", "56.2%")),
    NumericCheck("EXP-016 A-7d-M", "experiments/logs/exp-016.md", "A-7d-M", "A-7d-M", ("7", "65M", "1.02M", "19.31", "70.7%")),
    NumericCheck("EXP-016 B-14d-M", "experiments/logs/exp-016.md", "B-14d-M", "B-14d-M", ("14", "130M", "1.69M", "18.96", "65.8%")),
    NumericCheck("EXP-016 C-31d-M", "experiments/logs/exp-016.md", "C-31d-M", "C-31d-M", ("31", "262M", "3.04M", "19.39", "65.8%")),
    NumericCheck("EXP-016 D-62d-M", "experiments/logs/exp-016.md", "D-62d-M", "D-62d-M", ("62", "441M", "4.86M", "19.80", "68.1%")),
    # EXP-049 tokenizer sweep.
    NumericCheck("EXP-049 0.6B 4096 h64", "experiments/logs/exp-049.md", "exp049-0.6b-nc4096-h64", "0.6B nc4096 h64", ("0.67%", "0.3243", "0.3546", "0.0797")),
    NumericCheck("EXP-049 0.6B 8192 h128", "experiments/logs/exp-049.md", "exp049-0.6b-nc8192-h128", "0.6B nc8192 h128", ("0.42%", "0.3914", "0.2375", "0.0919")),
    NumericCheck("EXP-049 4B 4096 h128", "experiments/logs/exp-049.md", "exp049-4b-nc4096-h128", "4B nc4096 h128", ("1.60%", "0.2877", "0.3539", "0.1255")),
    NumericCheck("EXP-049 4B 8192 h128", "experiments/logs/exp-049.md", "exp049-4b-nc8192-h128", "4B nc8192 h128", ("1.28%", "0.3192", "0.2530", "0.1307")),
    NumericCheck("EXP-049 4B MGMR", "experiments/logs/exp-049.md", "exp049-4b-nc8192x2048-h128", "4B nc8192x2048 h128", ("1.41%", "0.3191", "0.3170", "0.1154")),
    # Full-eval baselines.
    NumericCheck("EXP-043 S-tier bare", "experiments/logs/exp-043.md", "exp043-s-0.6b", "S-tier bare", ("11.4%", "61.2%", "26.52")),
    NumericCheck("EXP-043 M-tier 0.6B", "experiments/logs/exp-043.md", "exp043-m-0.6b", "M-tier bare, 0.6B SID", ("14.5%", "70.2%", "18.54")),
    NumericCheck("EXP-043 M-tier 4B", "experiments/logs/exp-043.md", "exp043-m-4b", "M-tier, 4B SID", ("14.2%", "70.4%", "16.55")),
    NumericCheck("NTP README TO-RoPE", "experiments/logs/ntp/README.md", "S-tier with TO-RoPE", "S-tier with TO-RoPE", ("11.8%", "63.9%", "22.7")),
    NumericCheck("EXP-047 L-tier", "experiments/logs/exp-047.md", "exp047", "L-tier with validated options", ("12.8%", "64.1%", "20.7")),
    # EXP-029 alignment table.
    NumericCheck("EXP-029 on-policy", "experiments/logs/exp-029.md", "exp029-ecpo-onpolicy-w003-r100", "On-policy ECPO", ("13.0%", "67.8%", "14.1", "92%")),
    NumericCheck("EXP-029 off-policy", "experiments/logs/exp-029.md", "exp028-ecpo-weighted-w003-r100", "Off-policy ECPO", ("0.7%", "2.0%", "3791", "99%")),
    NumericCheck("EXP-029 SFT", "experiments/logs/exp-029.md", "exp020-hard-lam03", "SFT baseline", ("14.1%", "66.2%", "16.3")),
    # Public benchmark result.
    NumericCheck("MovieLens 1M Colab T4", "public_benchmarks/results/ml-1m-colab-t4.md", "ml-1m", "ml-1m", ("5,950", "3,532", "348,363", "10.5%", "40.4%", "72.5%", "85.2%")),
    NumericCheck("MovieLens 1M SID found", "public_benchmarks/results/ml-1m-colab-t4.md", "target_sid_found_rate", "target-SID found rate", ("89.9%",)),
]


def check_author(tex: str, failures: list[str]) -> None:
    forbidden = ["Author Name", "email@example.com", "Affiliation"]
    for token in forbidden:
        if token in tex:
            fail(f"author block still contains placeholder {token!r}", failures)
    for token in ["Ziqing Ye", "yeziqing986@gmail.com"]:
        if token not in tex:
            fail(f"author block missing {token!r}", failures)


def check_citations(tex: str, failures: list[str]) -> None:
    bib = BIB.read_text()
    cited = {
        key.strip()
        for match in re.finditer(r"\\cite[tp]?\{([^}]+)\}", tex)
        for key in match.group(1).split(",")
    }
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = sorted(cited - keys)
    if missing:
        fail(f"missing bibliography entries: {missing}", failures)


def check_figures(tex: str, failures: list[str]) -> None:
    for fig in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex):
        if not (PAPER / fig).exists():
            fail(f"missing figure file: {fig}", failures)


def check_experiment_count(tex: str, failures: list[str]) -> None:
    count = len([p for p in (ROOT / "experiments/logs").glob("exp-*.md") if not p.name.endswith(".zh.md")])
    if count != 51:
        fail(f"expected 51 English experiment logs, found {count}", failures)
    if "51 recorded experiments" not in tex:
        fail("paper no longer states '51 recorded experiments'", failures)


def check_numeric_tables(tex: str, failures: list[str]) -> None:
    source_cache: dict[str, str] = {}
    for check in NUMERIC_CHECKS:
        path = ROOT / check.source_path
        source = source_cache.setdefault(check.source_path, path.read_text())
        check_contains(source, check.source_token, list(check.values), f"{check.name} source", failures)
        check_contains(tex, check.tex_token, list(check.values), f"{check.name} manuscript", failures)


def main() -> int:
    failures: list[str] = []
    tex = TEX.read_text()

    check_author(tex, failures)
    check_citations(tex, failures)
    check_figures(tex, failures)
    check_experiment_count(tex, failures)
    check_numeric_tables(tex, failures)

    if failures:
        print("Paper consistency check failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Paper consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
