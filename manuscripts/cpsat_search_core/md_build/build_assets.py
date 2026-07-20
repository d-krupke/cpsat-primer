#!/usr/bin/env python3
"""
Asset generation for the Markdown backend of the Search Core manuscript.

Scans the LaTeX section files for the two float kinds that cannot be rendered as
text -- pseudocode `algorithm` blocks and `figure` blocks wrapping a graphic --
and turns each into an SVG image plus a caption, writing an `assets.json`
manifest that the later text-conversion stage consumes. SVG is used so the
pseudocode and diagrams stay crisp at any zoom on the mdbook site.

  * algorithm : the `algorithmic` body is wrapped in a standalone document that
                \\input's content_preamble.tex (so it renders identically to the
                PDF), compiled with xelatex, and converted to SVG (pdftocairo).
  * figure    : the already-existing referenced PDF is cropped (pdfcrop) and
                converted to SVG; no LaTeX compile is needed.

The float parser (find_floats) is written to be reused by the text stage, which
must strip these same blocks out and drop image references in their place.

Usage:
    python3 build_assets.py [--keep-tex]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parent
REPO = HERE.parents[2]  # .../cpsat-primer
SECTIONS_DIR = MANUSCRIPT / "sections"
PREAMBLE = HERE / "content_preamble.tex"
OUT_DIR = REPO / "images" / "search_core"  # committed alongside the primer images
BUILD_DIR = HERE / "_build"  # scratch for standalone .tex/.pdf

# a4paper, margin=2.5cm  ->  textwidth = 210mm - 2*25mm = 160mm.
# The pseudocode is rendered in a fixed-width box of exactly this width so that
# \algorithmiccomment's \hfill right-aligns the comments just as it does in print.
TEXTWIDTH = "160mm"

SECTION_ORDER = [
    "01-what-kind-of-solver.tex",
    "02-foundations.tex",
    "03-learning-from-failure.tex",
    "04-lazy-encoding.tex",
    "05-putting-it-together.tex",
    "06-going-faster.tex",
    "07-reflection.tex",
]


# --------------------------------------------------------------------------- #
# LaTeX brace / environment parsing (shared with the text stage later)
# --------------------------------------------------------------------------- #
def match_brace(text: str, open_pos: int) -> int:
    """Given index of a '{', return index just past the matching '}'."""
    assert text[open_pos] == "{"
    depth = 0
    i = open_pos
    while i < len(text):
        c = text[i]
        if c == "\\":  # skip escaped char (e.g. \{ \} )
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


def extract_command_arg(text: str, cmd: str) -> str | None:
    """Return the (balanced) first argument of the first \\cmd{...} in text."""
    m = re.search(r"\\" + cmd + r"\s*\{", text)
    if not m:
        return None
    open_pos = m.end() - 1
    close = match_brace(text, open_pos)
    return text[open_pos + 1 : close - 1].strip()


@dataclass
class Float:
    kind: str  # "algorithm" | "figure"
    section: str  # source filename
    number: int  # 1-based index within its kind, document order
    label: str | None
    caption: str | None  # raw LaTeX of the caption body
    body: str  # inner content (algorithmic body, or includegraphics args)
    graphic: str | None  # for figures: the referenced graphic path (rel. to manuscript)


def find_floats(text: str, section: str) -> list[Float]:
    """Find every algorithm/figure environment in `text`, in document order."""
    floats: list[Float] = []
    env_re = re.compile(r"\\begin\{(algorithm|figure)\}")
    for m in env_re.finditer(text):
        kind = m.group(1)
        end_tok = "\\end{" + kind + "}"
        end = text.find(end_tok, m.end())
        if end == -1:
            raise ValueError(f"unterminated {kind} in {section}")
        inner = text[m.end() : end]
        label = extract_command_arg(inner, "label")
        caption = extract_command_arg(inner, "caption")
        graphic = None
        body = inner
        if kind == "figure":
            graphic = extract_command_arg(inner, "includegraphics")
            # includegraphics may carry an optional [..] before the {path}
            gm = re.search(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]*)\}", inner)
            if gm:
                graphic = gm.group(1).strip()
        else:  # algorithm: capture the algorithmic environment verbatim
            am = re.search(
                r"\\begin\{algorithmic\}.*?\\end\{algorithmic\}", inner, re.DOTALL
            )
            body = am.group(0) if am else inner
        floats.append(
            Float(
                kind=kind,
                section=section,
                number=0,  # assigned after collecting all, per kind
                label=label,
                caption=caption,
                body=body.strip(),
                graphic=graphic,
            )
        )
    return floats


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
# hyperref+cleveref+xr let \cref inside pseudocode comments resolve to the SAME
# section numbers as the print build, by importing labels from the main
# document's .aux (produced by a normal `make` of the manuscript).
STANDALONE_TEMPLATE = r"""\documentclass[border=6pt]{standalone}
\input{%(preamble)s}
\usepackage[hidelinks]{hyperref}
\usepackage[capitalise,noabbrev]{cleveref}
\usepackage{xr}
\externaldocument{%(mainaux)s}
\begin{document}
\begin{minipage}{%(width)s}
%(body)s
\end{minipage}
\end{document}
"""


def run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def render_algorithm(fl: Float, keep_tex: bool) -> Path:
    """Compile one algorithm to a tightly-cropped SVG; return its path."""
    stem = fl.label.replace(":", "_") if fl.label else f"alg_{fl.number}"
    tex_path = BUILD_DIR / f"{stem}.tex"
    tex_path.write_text(
        STANDALONE_TEMPLATE
        % {
            "preamble": PREAMBLE.as_posix(),
            "mainaux": (MANUSCRIPT / "cpsat_search_core").as_posix(),
            "width": TEXTWIDTH,
            "body": fl.body,
        }
    )
    run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=BUILD_DIR,
    )
    pdf_path = BUILD_DIR / f"{stem}.pdf"  # standalone already crops tightly
    svg_path = OUT_DIR / f"{stem}.svg"
    run(["pdftocairo", "-svg", pdf_path.as_posix(), svg_path.as_posix()], cwd=BUILD_DIR)
    if not keep_tex:
        tex_path.unlink(missing_ok=True)
    return svg_path


def render_figure(fl: Float) -> Path:
    """Convert a figure's existing graphic (PDF) to a tightly-cropped SVG."""
    stem = fl.label.replace(":", "_") if fl.label else f"fig_{fl.number}"
    src = (MANUSCRIPT / fl.graphic).resolve()
    if not src.exists():
        raise SystemExit(f"figure graphic not found: {src}")
    svg_path = OUT_DIR / f"{stem}.svg"
    if src.suffix.lower() == ".pdf":
        # Ipe/TeX figure PDFs are exported full-page; crop to the drawing first.
        cropped = BUILD_DIR / f"{stem}_crop.pdf"
        run(
            ["pdfcrop", "--margins", "4", src.as_posix(), cropped.as_posix()],
            cwd=BUILD_DIR,
        )
        run(
            ["pdftocairo", "-svg", cropped.as_posix(), svg_path.as_posix()],
            cwd=BUILD_DIR,
        )
    else:  # already a raster: fall back to a straight copy
        svg_path = OUT_DIR / f"{stem}{src.suffix}"
        svg_path.write_bytes(src.read_bytes())
    return svg_path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keep-tex",
        action="store_true",
        help="keep the per-algorithm standalone .tex files for debugging",
    )
    args = ap.parse_args()

    # The pseudocode's \cref numbers are imported from the manuscript's .aux;
    # fail loudly if it is missing rather than silently rendering "??".
    if not (MANUSCRIPT / "cpsat_search_core.aux").exists():
        raise SystemExit(
            "cpsat_search_core.aux not found -- run `make` (build the PDF) first."
        )

    # Regenerate from scratch so a float removed from the LaTeX (or an output
    # whose format changed) leaves no stale file behind. OUT_DIR is a dedicated
    # folder owned by this pipeline, so clearing every file in it is safe.
    if OUT_DIR.exists():
        for old in OUT_DIR.iterdir():
            if old.is_file():
                old.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    all_floats: list[Float] = []
    for name in SECTION_ORDER:
        path = SECTIONS_DIR / name
        if not path.exists():
            continue
        all_floats.extend(find_floats(path.read_text(), name))

    # number per kind in document order
    counters: dict[str, int] = {}
    for fl in all_floats:
        counters[fl.kind] = counters.get(fl.kind, 0) + 1
        fl.number = counters[fl.kind]

    manifest = []
    for fl in all_floats:
        print(f"[{fl.kind} {fl.number}] {fl.label or '(no label)'}  <- {fl.section}")
        if fl.kind == "algorithm":
            img = render_algorithm(fl, args.keep_tex)
        else:
            img = render_figure(fl)
        rec = asdict(fl)
        rec["image"] = img.name  # filename only; consumers build the URL/path
        rec.pop("body")  # keep the manifest readable; body isn't needed downstream
        manifest.append(rec)

    (HERE / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {len(manifest)} assets -> {OUT_DIR}")
    print(f"manifest -> {HERE / 'assets.json'}")


if __name__ == "__main__":
    main()
