#!/usr/bin/env python3
"""
Text translator: LaTeX section sources -> primer-flavoured Markdown.

Deliberately NOT pandoc. The manuscript uses a small, known vocabulary of
commands (see the inventory in the design notes), so a bespoke converter is
safer and fully predictable. This stage consumes:

  * sections/*.tex            -- the content
  * assets.json               -- float -> image/caption/number (from build_assets.py)
  * cpsat_search_core.aux     -- \\cref numbers, identical to the print build
  * references.bib            -- bibliography

and emits one Markdown file per section plus references.md.

Conventions targeted (match existing primer chapters):
  * display math  \\[ ... \\]  ->  $$ ... $$
  * inline math   $ ... $      ->  \\( ... \\)   (mdbook needs \\( \\), not $)
  * callouts      platypus*    ->  > [!NOTE] / [!WARNING] / :reference: blocks
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parent
REPO = HERE.parents[2]  # .../cpsat-primer
MAIN_TEX = MANUSCRIPT / "cpsat_search_core.tex"
SECTIONS_DIR = MANUSCRIPT / "sections"
AUX = MANUSCRIPT / "cpsat_search_core.aux"
BIB = MANUSCRIPT / "references.bib"
MANIFEST = HERE / "assets.json"

ORSHA = "98c165af62df62b3056c2ee0fca66b24e79097cb"
ORBASE = f"https://github.com/google/or-tools/blob/{ORSHA}/ortools/sat/"
# primer images are referenced by absolute raw URL (same convention as the
# hand-written chapters); the assets were emitted into images/search_core/.
IMG_BASE = (
    "https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/search_core"
)
PDF_URL = (
    "https://github.com/d-krupke/cpsat-primer/blob/main/"
    "manuscripts/cpsat_search_core/cpsat_search_core.pdf"
)
CHAPTER = REPO / "chapters" / "search_core.md"

SECTION_ORDER = [
    "01-what-kind-of-solver.tex",
    "02-foundations.tex",
    "03-learning-from-failure.tex",
    "04-lazy-encoding.tex",
    "05-putting-it-together.tex",
    "06-going-faster.tex",
    "07-reflection.tex",
]

# cref label prefix -> printed noun (cleveref would print these)
CREF_NOUN = {"sec": "Section", "part": "Section", "fig": "Figure", "alg": "Algorithm"}


# --------------------------------------------------------------------------- #
# small LaTeX parsing helpers
# --------------------------------------------------------------------------- #
def match_brace(text: str, open_pos: int) -> int:
    depth = 0
    i = open_pos
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


def read_group(text: str, i: int) -> tuple[str, int]:
    """text[i] == '{'; return (inner, index past closing '}')."""
    end = match_brace(text, i)
    return text[i + 1 : end - 1], end


def unescape(s: str) -> str:
    """Undo LaTeX escapes in verbatim-ish content (\\_ -> _, etc.)."""
    return re.sub(r"\\([_&%#$~^{}])", r"\1", s)


# --------------------------------------------------------------------------- #
# reference data: aux numbers, label->file index, citations, bibliography
# --------------------------------------------------------------------------- #
def parse_aux(path: Path) -> dict[str, str]:
    """label -> printed number (e.g. 'sec:trace' -> '3.2')."""
    out: dict[str, str] = {}
    text = path.read_text()
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{", text):
        label = m.group(1)
        if label.endswith("@cref") or "@" in label:
            continue
        # first argument group after the label is {{number}...}
        grp, _ = read_group(text, m.end() - 1)
        nm = re.match(r"\s*\{([^{}]*)\}", grp)
        if nm:
            out[label] = nm.group(1).strip()
    return out


def md_name(section_file: str) -> str:
    return section_file.replace(".tex", ".md")


def build_label_index(sections: dict[str, str]) -> dict[str, str]:
    """label -> markdown filename where its anchor lives."""
    idx: dict[str, str] = {}
    for fname, text in sections.items():
        for m in re.finditer(r"\\label\{([^}]+)\}", text):
            idx[m.group(1)] = md_name(fname)
    return idx


def parse_citations(sections: dict[str, str]) -> dict[str, int]:
    """key -> citation number, by order of first appearance (sorting=none)."""
    order: dict[str, int] = {}
    for fname in SECTION_ORDER:
        text = sections.get(fname, "")
        for m in re.finditer(r"\\cite\{([^}]+)\}", text):
            for key in (k.strip() for k in m.group(1).split(",")):
                if key not in order:
                    order[key] = len(order) + 1
    return order


def parse_bib(path: Path) -> dict[str, dict[str, str]]:
    """Very small BibTeX field reader (author/title/year/... as flat strings)."""
    text = path.read_text()
    entries: dict[str, dict[str, str]] = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        key = m.group(2).strip()
        start = m.end()
        # fields until the entry's closing brace (assume balanced from '{' after key)
        brace_open = text.rfind("{", m.start(), m.end())
        entry_end = match_brace(text, brace_open)
        body = text[start:entry_end]
        fields: dict[str, str] = {"_type": m.group(1).lower()}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            fname = fm.group(1).lower()
            j = fm.end()
            if j < len(body) and body[j] == "{":
                val, _ = read_group(body, j)
            elif j < len(body) and body[j] == '"':  # quote-delimited field
                val = body[j + 1 : body.find('"', j + 1)]
            else:
                val = body[j:].split(",", 1)[0]
            fields[fname] = re.sub(r"\s+", " ", val).strip()
        entries[key] = fields
    return entries


# --------------------------------------------------------------------------- #
# math
# --------------------------------------------------------------------------- #
def convert_math(inner: str) -> str:
    """Expand the manuscript's custom math macros into plain MathJax."""

    # \bl{...} -> [\![ ... ]\!]   (double-bracket bound literal)
    def expand_bl(s: str) -> str:
        out, i = [], 0
        while i < len(s):
            m = re.match(r"\\bl\s*\{", s[i:])
            if m:
                grp, end = read_group(s, i + m.end() - 1)
                out.append(r"[\![" + expand_bl(grp) + r"]\!]")
                i = i + (end - i)
            else:
                out.append(s[i])
                i += 1
        return "".join(out)

    inner = expand_bl(inner)
    # \code{obj} inside math -> \texttt{obj}
    inner = re.sub(r"\\code\s*\{([^{}]*)\}", r"\\texttt{\1}", inner)
    # source-permalink macros can't be links inside math: keep just the name
    inner = re.sub(
        r"\\srcla\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\\texttt{\1}", inner
    )
    inner = re.sub(r"\\srcl\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\\texttt{\1:\2}", inner)
    inner = re.sub(r"\\srcf\s*\{([^{}]*)\}", r"\\texttt{\1}", inner)
    # MathJax has no \textsc; approximate with upright text
    inner = re.sub(r"\\textsc\s*\{([^{}]*)\}", r"\\text{\1}", inner)
    inner = inner.strip()
    # The mdbook markdown parser eats backslash-escapes before punctuation
    # (so `\!`, `\{`, `\}`, and the `\\` line breaks in `aligned` would be lost
    # before MathJax runs). Double every backslash: markdown collapses `\\`->`\`,
    # delivering the intended control sequence to MathJax intact.
    return inner.replace("\\", "\\\\")


# --------------------------------------------------------------------------- #
# inline conversion (the central walker: text + math + inline commands)
# --------------------------------------------------------------------------- #
class Converter:
    def __init__(self, aux, label_index, cites, current_md):
        self.aux = aux
        self.label_index = label_index
        self.cites = cites
        self.current_md = current_md

    # -- cross references & citations --------------------------------------
    def _ref_target(self, label: str) -> str:
        # single-file chapter: every anchor lives on the same page
        return "#" + label.replace(":", "-")

    def cref(self, label: str, cap: bool) -> str:
        labels = [x.strip() for x in label.split(",")]
        noun = CREF_NOUN.get(labels[0].split(":", 1)[0], "Section")
        if len(labels) == 1:
            num = self.aux.get(labels[0], "??")
            return f"[{noun} {num}]({self._ref_target(labels[0])})"
        # cleveref-style list: "Sections 5.1, 5.2 and 5.3"
        links = [f"[{self.aux.get(x, '??')}]({self._ref_target(x)})" for x in labels]
        joined = ", ".join(links[:-1]) + " and " + links[-1]
        return f"{noun}s {joined}"

    def crefrange(self, a: str, b: str) -> str:
        na, nb = self.aux.get(a, "??"), self.aux.get(b, "??")
        noun = CREF_NOUN.get(a.split(":", 1)[0], "Section") + "s"
        return f"[{noun} {na}–{nb}]({self._ref_target(a)})"

    def cite(self, keys: str) -> str:
        parts = []
        for key in (k.strip() for k in keys.split(",")):
            n = self.cites.get(key, "?")
            parts.append(f"[{n}](#ref-{key})")
        return "[" + ", ".join(parts) + "]"

    # -- source permalinks -------------------------------------------------
    def srcl(self, f, line):
        f = unescape(f)
        return f"[`{f}:{line}`]({ORBASE}{f}#L{line})"

    def srcla(self, f, line, text):
        f = unescape(f)
        return f"[{self.inline(text)}]({ORBASE}{f}#L{line})"

    def srcf(self, f):
        f = unescape(f)
        return f"[`{f}`]({ORBASE}{f})"

    # -- the walker --------------------------------------------------------
    def inline(self, text: str) -> str:
        out: list[str] = []
        i, n = 0, len(text)
        plain: list[str] = []

        def flush():
            if plain:
                out.append(normalize_text("".join(plain)))
                plain.clear()

        while i < n:
            c = text[i]
            if c == "%":  # stray comment (post strip this is rare)
                j = text.find("\n", i)
                i = n if j == -1 else j
                continue
            if c == "~":
                plain.append(" ")
                i += 1
                continue
            if c == "$":  # inline math -> keep as $...$ (build.py -> \(...\))
                flush()
                end = text.find("$", i + 1)
                # inline math must stay on ONE line: the primer's build.py converts
                # $...$ with a non-DOTALL regex, so a newline inside would break it.
                math = convert_math(text[i + 1 : end]).replace("\n", " ")
                out.append("$" + re.sub(r"\s+", " ", math) + "$")
                i = end + 1
                continue
            if text.startswith(r"\[", i):  # display math -> ```math fenced block
                flush()
                end = text.find(r"\]", i + 2)
                out.append(
                    "\n\n```math\n" + convert_math(text[i + 2 : end]) + "\n```\n\n"
                )
                i = end + 2
                continue
            if c == "{":  # bare group
                inner, end = read_group(text, i)
                out.append(self.inline(inner))
                i = end
                continue
            if c == "}":  # stray
                i += 1
                continue
            if c == "\\":
                flush()
                piece, i = self.command(text, i)
                out.append(piece)
                continue
            plain.append(c)
            i += 1
        flush()
        return "".join(out)

    def command(self, text: str, i: int) -> tuple[str, int]:
        m = re.match(r"\\([a-zA-Z]+)\*?", text[i:])
        if not m:  # escaped symbol: \_, \&, \%, \#, \$, \{, \}, \ (space), etc.
            ch = text[i + 1] if i + 1 < len(text) else ""
            if ch == " ":
                return " ", i + 2
            return ch, i + 2
        name = m.group(1)
        j = i + m.end()

        def arg():
            nonlocal j
            while j < len(text) and text[j] in " \t\n":
                j += 1
            g, end = read_group(text, j)
            j = end
            return g

        # commands taking arguments
        if name in ("textbf",):
            return f"**{self.inline(arg())}**", j
        if name in ("emph", "textit"):
            return f"_{self.inline(arg())}_", j
        if name in ("textsc", "textsf", "textnormal", "text", "mbox"):
            return self.inline(arg()), j
        if name == "code":
            return f"`{unescape(arg())}`", j
        if name == "href":
            url, txt = arg(), arg()
            return f"[{self.inline(txt)}]({url})", j
        if name == "srcl":
            return self.srcl(arg(), arg()), j
        if name == "srcla":
            return self.srcla(arg(), arg(), arg()), j
        if name == "srcf":
            return self.srcf(arg()), j
        if name in ("cref", "Cref"):
            return self.cref(arg(), name[0].isupper()), j
        if name == "crefrange":
            return self.crefrange(arg(), arg()), j
        if name == "cite":
            return self.cite(arg()), j
        if name == "label":
            arg()  # consumed; anchors are emitted at block level
            return "", j
        # zero-arg tokens
        ZERO = {
            "orsha": ORSHA,
            "dots": "…",
            "ldots": "…",
            "quad": " ",
            "qquad": " ",
            " ": " ",
            ",": " ",
            ";": " ",
            ":": "",
            "%": "%",
            "&": "&",
            "#": "#",
            "_": "_",
            "LaTeX": "LaTeX",
            "TeX": "TeX",
        }
        if name in ZERO:
            return ZERO[name], j
        # unknown command: drop it, keep any following group as text
        return "", j


def normalize_text(s: str) -> str:
    s = s.replace("``", "“").replace("''", "”")
    s = s.replace("---", "—").replace("--", "–")
    return s


# --------------------------------------------------------------------------- #
# block-level conversion
# --------------------------------------------------------------------------- #
# platypus env -> (github callout token, fixed title or None if title is optional)
CALLOUT = {
    "platypusinfo": ("[!NOTE]", None),
    "platypustip": ("[!TIP]", None),
    "platypuswarning": ("[!WARNING]", None),
    "platypusbook": (":reference:", "Background"),
    "platypuslog": (":log:", "In the solve log"),
    "platypusparam": (":tune:", "Tuning the search"),
}
BLOCK_ENVS = tuple(CALLOUT) + ("itemize", "enumerate", "quote")


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def extract_floats(text: str) -> str:
    """Replace algorithm/figure environments with @@FLOAT:label@@ tokens."""

    def repl(m):
        inner = m.group(2)
        lm = re.search(r"\\label\{([^}]+)\}", inner)
        return f"\n\n@@FLOAT:{lm.group(1)}@@\n\n" if lm else "\n\n"

    return re.sub(
        r"\\begin\{(algorithm|figure)\}(.*?)\\end\{\1\}", repl, text, flags=re.DOTALL
    )


def opt_args(text: str, i: int) -> int:
    """Skip a run of optional [..] arguments starting at i; return new index."""
    while i < len(text) and text[i] == "[":
        depth, i = 0, i
        while i < len(text):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        while i < len(text) and text[i] in " \t\n":
            i += 1
    return i


def find_env_end(text: str, env: str, start: int) -> int:
    """Index just past the matching \\end{env}, honouring nested same-name envs."""
    depth, i = 1, start
    b = re.compile(r"\\begin\{" + env + r"\}")
    e = re.compile(r"\\end\{" + env + r"\}")
    while i < len(text):
        mb, me = b.search(text, i), e.search(text, i)
        if me is None:
            raise ValueError(f"unterminated {env}")
        if mb and mb.start() < me.start():
            depth += 1
            i = mb.end()
        else:
            depth -= 1
            i = me.end()
            if depth == 0:
                return i
    raise ValueError(f"unterminated {env}")


class BlockRenderer:
    def __init__(self, conv: Converter, floats: dict[str, dict]):
        self.c = conv
        self.floats = floats
        self.sec_no = 0  # running \section number (article-style 1..n)
        self.sub_no = 0  # running \subsection number within the current section

    def paragraphs(self, text: str) -> str:
        chunks = re.split(r"\n\s*\n", text)
        out = []
        for ch in chunks:
            md = self.c.inline(ch).strip()
            if md:
                out.append(md)
        return "\n\n".join(out)

    def float_md(self, label: str) -> str:
        rec = self.floats.get(label)
        anchor = f'<a name="{label.replace(":", "-")}"></a>'
        if not rec:
            return anchor
        img = f"{IMG_BASE}/{rec['image']}"
        noun = "Algorithm" if rec["kind"] == "algorithm" else "Figure"
        cap = self.c.inline(rec["caption"] or "").strip()
        # Render at the full text width, mirroring \includegraphics[width=\linewidth]
        # in the PDF (markdown image syntax can't set a width, so use <img>).
        return (
            f"{anchor}\n\n"
            f'<img src="{img}" alt="{noun} {rec["number"]}" style="width: 100%;">\n\n'
            f"**{noun} {rec['number']}.** {cap}"
        )

    def list_md(self, env: str, inner: str, depth: int) -> str:
        items = re.split(r"\\item\b", inner)
        items = [it for it in items[1:]]  # text before first \item is empty
        lines = []
        for n, it in enumerate(items, 1):
            marker = f"{n}. " if env == "enumerate" else "- "
            body = self.render(it.strip(), depth + 1).strip()
            body_lines = body.split("\n")
            pad = "  " * depth
            lines.append(pad + marker + body_lines[0])
            for extra in body_lines[1:]:
                lines.append((pad + "  " + extra) if extra.strip() else "")
        return "\n".join(lines)

    def callout_md(self, env: str, inner: str, title_override) -> str:
        token, fixed_title = CALLOUT[env]
        title = title_override or fixed_title
        body = self.render(inner.strip(), 0).strip()
        lines = [f"> {token}"]
        if title:
            lines.append(f"> **{title}**")
            lines.append(">")
        for ln in body.split("\n"):
            lines.append(f"> {ln}" if ln.strip() else ">")
        return "\n".join(lines)

    def heading(
        self, level: int, title: str, rest_after: str, number: str
    ) -> tuple[str, str]:
        """Return (markdown heading, remaining text). Consumes a trailing \\label."""
        rest = rest_after.lstrip()
        anchor = ""
        lm = re.match(r"\\label\{([^}]+)\}", rest)
        if lm:
            anchor = f'<a name="{lm.group(1).replace(":", "-")}"></a>\n\n'
            rest = rest[lm.end() :]
        hashes = "#" * level
        return f"{anchor}{hashes} {number} {self.c.inline(title).strip()}", rest

    def render(self, text: str, depth: int) -> str:
        """Convert a block of LaTeX body into Markdown, recursively."""
        out: list[str] = []
        i = 0
        # scan for the next structural construct; everything before it is prose
        pat = re.compile(
            r"\\(section|subsection)\{"
            r"|\\begin\{(" + "|".join(BLOCK_ENVS) + r")\}"
            r"|@@FLOAT:([^@]+)@@"
        )
        while i < len(text):
            m = pat.search(text, i)
            if not m:
                out.append(self.paragraphs(text[i:]))
                break
            if m.start() > i:
                out.append(self.paragraphs(text[i : m.start()]))

            if m.group(1):  # \section / \subsection (nested under the ## chapter title)
                if m.group(1) == "section":
                    self.sec_no += 1
                    self.sub_no = 0
                    level, number = 3, str(self.sec_no)
                else:
                    self.sub_no += 1
                    level, number = 4, f"{self.sec_no}.{self.sub_no}"
                title, end = read_group(text, m.end() - 1)
                heading, rest = self.heading(level, title, text[end:], number)
                out.append(heading)
                text, i = rest, 0
                continue

            if m.group(2):  # a block environment
                env = m.group(2)
                body_start = opt_args(text, m.end())
                end = find_env_end(text, env, body_start)
                inner = text[body_start : end - len(f"\\end{{{env}}}")]
                if env in CALLOUT:
                    # \begin{env}[placement][Title] -- a 2nd optional arg is a title
                    title = self._callout_title(text, m.end())
                    out.append(self.callout_md(env, inner, title))
                elif env in ("itemize", "enumerate"):
                    out.append(self.list_md(env, inner, depth))
                else:  # quote
                    q = self.render(inner.strip(), 0).strip()
                    out.append(
                        "\n".join(
                            f"> {ln}" if ln.strip() else ">" for ln in q.split("\n")
                        )
                    )
                i = end
                continue

            if m.group(3):  # @@FLOAT:label@@
                out.append(self.float_md(m.group(3)))
                i = m.end()
                continue

        return "\n\n".join(s for s in out if s.strip())

    def _callout_title(self, text: str, after_begin: int) -> str | None:
        """A titled callout is \\begin{env}[placement][Title]; return Title or None."""
        j = after_begin
        brackets = []
        while j < len(text) and text[j] == "[":
            depth, k = 0, j
            while k < len(text):
                if text[k] == "[":
                    depth += 1
                elif text[k] == "]":
                    depth -= 1
                    if depth == 0:
                        brackets.append(text[j + 1 : k])
                        k += 1
                        break
                k += 1
            j = k
            while j < len(text) and text[j] in " \t\n":
                j += 1
        return brackets[1] if len(brackets) >= 2 else None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def format_reference(key: str, num: int, fields: dict[str, str]) -> str:
    def clean(s):
        return (
            s.replace("{", "")
            .replace("}", "")
            .replace("\\'e", "é")
            .replace("\\'E", "É")
            .replace("\\'o", "ó")
            .replace("~", " ")
            .strip()
        )

    author = clean(fields.get("author", "")).replace(" and ", ", ")
    title = clean(fields.get("title", ""))
    year = fields.get("year", "")
    venue = clean(fields.get("booktitle", fields.get("journal", "")))
    url = fields.get("url", "")
    # Use HTML (not markdown) for emphasis/links: markdown syntax is NOT processed
    # inside a raw block-level <ol> in mdbook, so `*title*`/`[url](url)` would show
    # their literal characters. <em>/<a> render correctly there.
    bits = [b for b in [author, f"<em>{title}</em>", venue, year] if b]
    entry = ". ".join(b.rstrip(".") for b in bits)
    if url:
        entry += f'. <a href="{url}">{url}</a>'
    # The enclosing <ol> supplies the number (matching the citation order), so
    # the entry itself must NOT repeat it -- otherwise it reads "1. [1] ...".
    return f'<li id="ref-{key}">{entry}.</li>'


def extract_abstract(conv: Converter) -> str:
    tex = MAIN_TEX.read_text()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    if not m:
        return ""
    body = strip_comments(m.group(1)).replace(r"\noindent", "")
    return "\n\n".join(
        conv.inline(p).strip() for p in re.split(r"\n\s*\n", body) if p.strip()
    )


# Cover image (website only, like the other chapters): a platypus watchmaker
# peering into the opened engine of the solver. Lives in images/ (not the
# search_core/ subfolder) alongside the other chapter covers.
COVER_URL = "https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_search_core.webp"

CHAPTER_HEADER = f"""<!-- This chapter is GENERATED from manuscripts/cpsat_search_core \
by md_build/latex_to_md.py. Do not edit by hand; edit the LaTeX and regenerate. -->

<a name="search-core"></a>

## How CP-SAT Reasons: The Search Core

<!-- START_SKIP_FOR_README -->

![Cover Image Search Core]({COVER_URL})

<!-- STOP_SKIP_FOR_README -->
"""


def main() -> None:
    if not AUX.exists():
        raise SystemExit(
            "cpsat_search_core.aux not found -- run `make` (build the PDF) first."
        )
    if not MANIFEST.exists():
        raise SystemExit(
            "assets.json not found -- run build_assets.py first (make markdown does both)."
        )

    sections = {
        f: strip_comments((SECTIONS_DIR / f).read_text())
        for f in SECTION_ORDER
        if (SECTIONS_DIR / f).exists()
    }

    aux = parse_aux(AUX)
    label_index = build_label_index(sections)
    cites = parse_citations(sections)
    bib = parse_bib(BIB)
    manifest = {rec["label"]: rec for rec in json.loads(MANIFEST.read_text())}

    conv = Converter(aux, label_index, cites, "search_core.md")
    renderer = BlockRenderer(conv, manifest)

    # body: all sections concatenated into a single page
    bodies = []
    for fname, raw in sections.items():
        md = renderer.render(extract_floats(raw), 0)
        bodies.append(re.sub(r"\n{3,}", "\n\n", md).strip())
    body = "\n\n".join(bodies)

    # references as an HTML list (real #ref-<key> anchors for the site)
    items = [format_reference(k, n, bib.get(k, {})) for k, n in cites.items()]
    references = "### References\n\n<ol>\n" + "\n".join(items) + "\n</ol>"

    pdf_note = (
        "> [!NOTE]\n"
        "> This is an in-depth companion chapter. It is also available as a "
        f"**[typeset PDF]({PDF_URL})** with full typography, and its content is "
        "generated from that LaTeX source.\n"
    )

    chapter = "\n\n".join(
        [
            CHAPTER_HEADER.strip(),
            extract_abstract(conv),
            pdf_note,
            "<!-- START_SKIP_FOR_README -->",
            body,
            references,
            "<!-- STOP_SKIP_FOR_README -->",
        ]
    )
    chapter = re.sub(r"\n{3,}", "\n\n", chapter).strip() + "\n"
    CHAPTER.write_text(chapter)
    print(f"wrote {CHAPTER}  ({len(chapter)} chars, {len(cites)} refs)")


if __name__ == "__main__":
    main()
