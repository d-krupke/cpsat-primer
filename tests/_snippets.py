"""Find and run the code blocks of the chapters that are marked for testing.

Created in November 2026 so tests exercise the snippets readers actually see,
instead of hand-maintained copies that drift from the book (PR #82 fixed
several snippets whose copies kept passing).

Marking a snippet
-----------------
Put an HTML comment on the line(s) above a python block in ``chapters/*.md``.
It is invisible on GitHub, in mdBook and in the generated README::

    <!-- snippet: modelling/abs_min_max -->
    ```python
    model = cp_model.CpModel()
    ...
    ```

Blank lines between marker and fence are fine (Prettier inserts one). The id
must start with the chapter's file name. Optional ``key=value`` options:

* ``requires=a/x,b/y`` -- run these snippets first, in the same namespace
  (e.g. a class that needs a data class defined in an earlier block).
* ``mode=compile`` -- only check that the block compiles; for sketches with
  placeholders or examples that need unavailable solvers.

Fragments that assume context the reader already has ("``x`` is defined in
your model") get hidden setup lines inside the comment. They run right before
the block and should state no more than the prose assumes::

    <!-- snippet: modelling/abs_objective
    model = cp_model.CpModel()
    x = model.new_int_var(-100, 100, "x")
    -->

Every marked snippet is smoke-tested by ``test_documented_snippets.py``. To
check what a snippet *computes*, write a test that calls
``run_snippet("modelling/abs_min_max")`` and asserts on the returned namespace.
Snippets run with ``cp_model`` predefined, as the chapters assume that import.
"""

import re
import textwrap
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from types import CodeType
from typing import Any, Literal

from ortools.sat.python import cp_model

CHAPTERS = Path(__file__).resolve().parents[1] / "chapters"

_MARKER = re.compile(r"^\s*<!--\s*snippet:\s*(?P<body>.*?)\s*-->\s*$")
_MARKER_OPEN = re.compile(r"^\s*<!--\s*snippet:\s*(?P<body>.*?)\s*$")
_OPEN_FENCE = re.compile(r"^(?P<indent>\s*)```(?P<lang>\S*)\s*$")
_CLOSE_FENCE = re.compile(r"^\s*```\s*$")
_MODES = ("run", "compile")


class SnippetError(ValueError):
    """A marker is malformed or the snippets do not form a valid set."""


@dataclass(frozen=True)
class Snippet:
    id: str
    code: str
    path: Path
    line: int  # 1-based markdown line of the first code line
    requires: tuple[str, ...] = ()
    mode: Literal["run", "compile"] = "run"
    setup: str = ""  # hidden lines from the marker, run before the code
    setup_line: int = 0

    def compile(self) -> CodeType:
        """Compile setup and code so that tracebacks show the chapter lines."""
        source = ""
        for line, code in ((self.setup_line, self.setup), (self.line, self.code)):
            if code:
                source += "\n" * (line - 1 - source.count("\n")) + code
        return compile(source, str(self.path), "exec")


def parse_markdown(text: str, path: Path) -> list[Snippet]:
    """Extract all marked python blocks of one chapter (pure, no file access)."""
    lines = text.splitlines()
    snippets = []
    for i, line in enumerate(lines):
        where = f"{path}:{i + 1}"
        if match := _MARKER.match(line):
            end = i  # single-line marker
        elif match := _MARKER_OPEN.match(line):
            end = next((j for j in range(i + 1, len(lines)) if "-->" in lines[j]), None)
            if end is None or lines[end].strip() != "-->":
                raise SnippetError(f"{where}: close the marker with a line '-->'")
        else:
            continue
        snippet_id, options = _parse_marker(match["body"], path.stem, where)
        snippet = _read_block(lines, end + 1, path, snippet_id, options, where)
        if end > i:
            setup = textwrap.dedent("".join(f"{s}\n" for s in lines[i + 1 : end]))
            snippet = replace(snippet, setup=setup, setup_line=i + 2)
        snippets.append(snippet)
    return snippets


def _parse_marker(body: str, chapter: str, where: str) -> tuple[str, dict[str, str]]:
    if not body:
        raise SnippetError(f"{where}: marker has no snippet id")
    snippet_id, *pairs = body.split()
    if not snippet_id.startswith(f"{chapter}/"):
        raise SnippetError(f"{where}: id {snippet_id!r} must start with '{chapter}/'")
    options = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SnippetError(f"{where}: expected key=value, got {pair!r}")
        if key not in ("requires", "mode"):
            raise SnippetError(f"{where}: unknown option {key!r}")
        options[key] = value
    if options.get("mode", "run") not in _MODES:
        raise SnippetError(f"{where}: mode must be one of {_MODES}")
    return snippet_id, options


def _read_block(lines, start, path, snippet_id, options, where) -> Snippet:
    while start < len(lines) and not lines[start].strip():
        start += 1
    fence = _OPEN_FENCE.match(lines[start]) if start < len(lines) else None
    if fence is None or fence["lang"] != "python":
        raise SnippetError(f"{where}: marker is not followed by a python block")
    end = start + 1
    while end < len(lines) and not _CLOSE_FENCE.match(lines[end]):
        end += 1
    if end == len(lines):
        raise SnippetError(f"{where}: python block is never closed")
    body = "".join(line + "\n" for line in lines[start + 1 : end])
    requires = tuple(r for r in options.get("requires", "").split(",") if r)
    return Snippet(
        id=snippet_id,
        code=textwrap.dedent(body),
        path=path,
        line=start + 2,
        requires=requires,
        mode=options.get("mode", "run"),
    )


def index_snippets(snippets: list[Snippet]) -> dict[str, Snippet]:
    """Key snippets by id and check that duplicates and requirements are sound."""
    index: dict[str, Snippet] = {}
    for snippet in snippets:
        if snippet.id in index:
            other = index[snippet.id]
            raise SnippetError(
                f"{snippet.path}:{snippet.line}: duplicate id {snippet.id!r}, "
                f"first used at {other.path}:{other.line}"
            )
        index[snippet.id] = snippet
    for snippet in index.values():
        for required in snippet.requires:
            if required not in index:
                raise SnippetError(
                    f"{snippet.id} requires unknown snippet {required!r}"
                )
            if index[required].mode == "compile":
                raise SnippetError(f"{snippet.id} requires compile-only {required!r}")
        execution_order(index, snippet.id)  # raises on cycles
    return index


def execution_order(index: dict[str, Snippet], snippet_id: str) -> list[Snippet]:
    """Requirements depth-first, each once, ending with the snippet itself."""
    order: list[Snippet] = []
    active: list[str] = []

    def visit(current: str) -> None:
        if current in active:
            raise SnippetError(f"requirement cycle: {' -> '.join([*active, current])}")
        if any(s.id == current for s in order):
            return
        active.append(current)
        for required in index[current].requires:
            visit(required)
        active.pop()
        order.append(index[current])

    visit(snippet_id)
    return order


@cache
def documented_snippets() -> dict[str, Snippet]:
    """All marked snippets of the book, validated."""
    snippets = []
    for path in sorted(CHAPTERS.glob("*.md")):
        # Absolute paths let tracebacks show the chapter line from any cwd.
        snippets += parse_markdown(path.read_text(encoding="utf-8"), path)
    return index_snippets(snippets)


def run_snippet(snippet_id: str, **predefined: Any) -> dict[str, Any]:
    """Run a snippet after its requirements and return the resulting namespace.

    ``predefined`` names are set before anything runs, e.g. ``__file__`` for
    snippets that write next to the file they assume to live in.
    """
    snippets = documented_snippets()
    if snippets[snippet_id].mode == "compile":
        raise SnippetError(f"{snippet_id} is compile-only and cannot be run")
    namespace: dict[str, Any] = {"cp_model": cp_model, **predefined}
    for snippet in execution_order(snippets, snippet_id):
        exec(snippet.compile(), namespace)  # noqa: S102 -- running the book is the point
    return namespace
