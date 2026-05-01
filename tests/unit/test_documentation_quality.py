"""Documentation and public API quality checks."""

from __future__ import annotations

import ast
from pathlib import Path


def _public_nodes(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            yield node, node.name
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("_"):
                    yield sub, f"{node.name}.{sub.name}"
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            yield node, node.name


def _has_full_hints(node: ast.FunctionDef) -> bool:
    args = [*node.args.args, *node.args.kwonlyargs]
    if node.args.vararg:
        args.append(node.args.vararg)
    if node.args.kwarg:
        args.append(node.args.kwarg)
    return node.returns is not None and all(
        arg.arg == "self" or arg.annotation is not None for arg in args
    )


def test_public_api_docstrings_and_type_hints() -> None:
    """DOC.1-DOC.3: public source APIs have docstrings and type hints."""
    missing_docstrings: list[str] = []
    missing_hints: list[str] = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, qualname in _public_nodes(tree):
            if ast.get_docstring(node) is None:
                missing_docstrings.append(f"{path}:{node.lineno}:{qualname}")
            if isinstance(node, ast.FunctionDef) and not _has_full_hints(node):
                missing_hints.append(f"{path}:{node.lineno}:{qualname}")
    assert missing_docstrings == []
    assert missing_hints == []
