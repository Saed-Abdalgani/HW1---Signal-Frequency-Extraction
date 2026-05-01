"""Count code lines (excluding blanks, comment-only, docstrings)."""
from pathlib import Path


def count_code_lines(filepath):
    lines = Path(filepath).read_text(encoding="utf-8").splitlines()
    in_docstring = False
    code = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Toggle docstring state
        triple = stripped.count('"""')
        if triple >= 2:
            continue  # single-line docstring
        if triple == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped.startswith("#"):
            continue
        code += 1
    return code, len(lines)

files = [
    "src/freq_extractor/sdk/sdk.py",
    "src/freq_extractor/sdk/sdk_runner.py",
    "src/freq_extractor/sdk/__init__.py",
    "src/main.py",
]

for f in files:
    code, total = count_code_lines(f)
    status = "OK" if code <= 145 else "OVER"
    print(f"  {status}  {f}: {code} code lines / {total} total")
