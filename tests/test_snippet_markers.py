"""Unit tests for the snippet-marker parser in `_snippets.py`.

These pin down the marker grammar that chapters rely on, using small inline
markdown documents instead of the real chapters, so a parser change cannot
hide behind a chapter edit.
"""

import traceback
from pathlib import Path

import pytest
from _snippets import SnippetError, execution_order, index_snippets, parse_markdown

CHAPTER = Path("chapters/demo.md")


def md(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def test_marked_block_is_extracted_with_its_line():
    """The id, code and the markdown line of the first code line are kept."""
    text = md(
        "Some prose.",
        "",
        "<!-- snippet: demo/first -->",
        "```python",
        "x = 1",
        "y = x + 1",
        "```",
    )
    (snippet,) = parse_markdown(text, CHAPTER)
    assert snippet.id == "demo/first"
    assert snippet.code == "x = 1\ny = x + 1\n"
    assert snippet.line == 5
    assert snippet.mode == "run"
    assert snippet.requires == ()


def test_unmarked_blocks_are_ignored():
    """Only marked blocks are snippets; the rest of the book stays untouched."""
    text = md("```python", "x = 1", "```", "<!-- unrelated -->", "```python", "```")
    assert parse_markdown(text, CHAPTER) == []


def test_options_and_blank_line_before_fence():
    """Formatters may insert a blank line between marker and fence."""
    text = md(
        "<!-- snippet: demo/second requires=demo/first,other/setup mode=compile -->",
        "",
        "```python",
        "z = 3",
        "```",
    )
    (snippet,) = parse_markdown(text, CHAPTER)
    assert snippet.requires == ("demo/first", "other/setup")
    assert snippet.mode == "compile"


def test_indented_fence_is_dedented():
    """Blocks nested in list items are indented in the markdown, not in Python."""
    text = md(
        "- item", "  <!-- snippet: demo/nested -->", "  ```python", "  a = 1", "  ```"
    )
    (snippet,) = parse_markdown(text, CHAPTER)
    assert snippet.code == "a = 1\n"


def test_compiled_code_reports_markdown_lines():
    """Tracebacks must point at the chapter line, not at a line in a temp string."""
    text = md("", "<!-- snippet: demo/fails -->", "```python", "ok = 1", "1 / 0", "```")
    (snippet,) = parse_markdown(text, CHAPTER)
    with pytest.raises(ZeroDivisionError) as info:
        exec(snippet.compile(), {})  # noqa: S102
    frame = traceback.extract_tb(info.value.__traceback__)[-1]
    assert (frame.filename, frame.lineno) == (str(CHAPTER), 5)


@pytest.mark.parametrize(
    "text, message",
    [
        (md("<!-- snippet: demo/a -->", "text"), "not followed by a python block"),
        (md("<!-- snippet: demo/a -->", "```json", "{}", "```"), "python block"),
        (md("<!-- snippet: demo/a -->", "```python", "x = 1"), "never closed"),
        (md("<!-- snippet: wrong/a -->", "```python", "```"), "must start with"),
        (md("<!-- snippet: demo/a color=red -->", "```python", "```"), "unknown"),
        (md("<!-- snippet: demo/a mode=fast -->", "```python", "```"), "mode"),
        (md("<!-- snippet: demo/a mode -->", "```python", "```"), "key=value"),
        (md("<!-- snippet: -->", "```python", "```"), "no snippet id"),
    ],
)
def test_malformed_markers_fail_loudly(text, message):
    """A typo in a marker must never silently drop a snippet from testing."""
    with pytest.raises(SnippetError, match=message):
        parse_markdown(text, CHAPTER)


def test_index_rejects_duplicates_and_unknown_requirements():
    duplicated = parse_markdown(
        md("<!-- snippet: demo/a -->", "```python", "```") * 2, CHAPTER
    )
    with pytest.raises(SnippetError, match="duplicate"):
        index_snippets(duplicated)
    dangling = parse_markdown(
        md("<!-- snippet: demo/a requires=demo/missing -->", "```python", "```"),
        CHAPTER,
    )
    with pytest.raises(SnippetError, match="unknown snippet"):
        index_snippets(dangling)


def test_index_rejects_cycles_and_compile_only_requirements():
    cyclic = parse_markdown(
        md(
            "<!-- snippet: demo/a requires=demo/b -->",
            "```python",
            "```",
            "<!-- snippet: demo/b requires=demo/a -->",
            "```python",
            "```",
        ),
        CHAPTER,
    )
    with pytest.raises(SnippetError, match="cycle"):
        index_snippets(cyclic)
    compile_only = parse_markdown(
        md(
            "<!-- snippet: demo/sketch mode=compile -->",
            "```python",
            "```",
            "<!-- snippet: demo/user requires=demo/sketch -->",
            "```python",
            "```",
        ),
        CHAPTER,
    )
    with pytest.raises(SnippetError, match="compile-only"):
        index_snippets(compile_only)


def test_execution_order_runs_requirements_first_and_once():
    """A diamond of requirements runs the shared base exactly once, first."""
    snippets = index_snippets(
        parse_markdown(
            md(
                "<!-- snippet: demo/base -->",
                "```python",
                "```",
                "<!-- snippet: demo/left requires=demo/base -->",
                "```python",
                "```",
                "<!-- snippet: demo/right requires=demo/base -->",
                "```python",
                "```",
                "<!-- snippet: demo/top requires=demo/left,demo/right -->",
                "```python",
                "```",
            ),
            CHAPTER,
        )
    )
    order = [s.id for s in execution_order(snippets, "demo/top")]
    assert order == ["demo/base", "demo/left", "demo/right", "demo/top"]
