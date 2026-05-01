"""Tests for UIService — 6K.1, 6K.2.

Validates Dash app construction and layout structure.
"""

from __future__ import annotations

import ast
from pathlib import Path

import dash

from freq_extractor.services.ui_service import UIService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_FILES = [
    "ui_service.py",
    "ui_layout.py",
    "ui_components.py",
    "ui_callbacks.py",
    "ui_analysis.py",
]


def _walk(component):
    """Yield every Dash component in a layout tree."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, list | tuple):
        children = [children]
    for child in children:
        yield from _walk(child)


def _ids(layout) -> set[str]:
    """Collect component IDs from a Dash layout tree."""
    return {c.id for c in _walk(layout) if hasattr(c, "id") and c.id is not None}


def _code_lines(path: Path) -> int:
    """Count non-comment source lines, excluding docstring lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    ignored: set[int] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        body = getattr(node, "body", [])
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            ignored.update(range(body[0].lineno, body[0].end_lineno + 1))
    return sum(
        1 for i, line in enumerate(lines, start=1)
        if i not in ignored and line.strip() and not line.lstrip().startswith("#")
    )


class TestUIService:
    """UIService tests."""

    def test_build_app_returns_dash(self, sample_config) -> None:
        """6K.1: build_app() returns a Dash app instance."""
        svc = UIService(config=sample_config)
        app = svc.build_app()
        assert isinstance(app, dash.Dash)

    def test_app_title(self, sample_config) -> None:
        """App title is 'SINUSOID EXPLORER'."""
        svc = UIService(config=sample_config)
        app = svc.build_app()
        assert app.title == "SINUSOID EXPLORER"

    def test_layout_not_none(self, sample_config) -> None:
        """App layout is set after build."""
        svc = UIService(config=sample_config)
        app = svc.build_app()
        assert app.layout is not None

    def test_default_config(self) -> None:
        """UIService works with no config (defaults)."""
        svc = UIService()
        app = svc.build_app()
        assert isinstance(app, dash.Dash)

    def test_build_idempotent(self, sample_config) -> None:
        """Calling build_app twice works without error."""
        svc = UIService(config=sample_config)
        svc.build_app()
        app2 = svc.build_app()
        assert isinstance(app2, dash.Dash)

    def test_four_tabs_present(self, sample_config) -> None:
        """8.16: Layout exposes the required four dashboard tabs."""
        app = UIService(config=sample_config).build_app()
        tabs = next(c for c in _walk(app.layout) if getattr(c, "id", None) == "main-tabs")
        labels = [child.label for child in tabs.children]
        assert labels == ["SIGNALS", "T-SNE 3D", "PCA 3D", "FFT SPECTRUM"]

    def test_per_sinusoid_controls_present(self, sample_config) -> None:
        """8.17: Each sinusoid block has MIX/BPF/f/phi/A/sigma controls."""
        app = UIService(config=sample_config).build_app()
        ids = _ids(app.layout)
        for n in range(1, 5):
            expected = {f"mix-{n}", f"bpf-{n}", f"freq-{n}", f"phase-{n}",
                        f"amp-{n}", f"sigma-{n}"}
            assert expected <= ids

    def test_ui_files_under_line_limit(self) -> None:
        """8.15: UI Python modules stay within the 145-code-line limit."""
        service_dir = PROJECT_ROOT / "src" / "freq_extractor" / "services"
        counts = {name: _code_lines(service_dir / name) for name in UI_FILES}
        assert all(lines <= 145 for lines in counts.values()), counts
