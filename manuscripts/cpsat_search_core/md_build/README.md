# Markdown backend

Generates the primer chapter [`chapters/search_core.md`](../../../chapters/search_core.md)
and its images automatically from the LaTeX source in this manuscript. The goal
is a single source of truth: write the manuscript in LaTeX, and the primer's
web/Markdown version is derived from it.

## The pipeline

Run from the manuscript root (`manuscripts/cpsat_search_core/`):

```sh
make markdown    # LaTeX -> chapters/search_core.md + images/search_core/*.svg
make primer      # ... then regenerate the primer README.md and .mdbook/*.md
make preview     # ... then `mdbook build` and open a self-contained local preview
```

Each target builds on the previous one. The usual loop while editing the
manuscript is `make preview` (does everything and opens the result); when you are
happy, commit the regenerated `chapters/search_core.md` and
`images/search_core/*.svg` — the primer's own `build.py` (run by pre-commit and
CI) turns them into the README and the website, and needs no LaTeX.

The two stages can also be run directly:

```sh
python3 md_build/build_assets.py     # 1. floats -> images/search_core/*.svg (+ assets.json)
python3 md_build/latex_to_md.py      # 2. sections/*.tex -> chapters/search_core.md
python3 md_build/preview.py          # (after mdbook build) open the inlined preview
```

Both stages need the manuscript's `.aux` (for cross-reference numbers), so build
the PDF first with `make`; they fail with a clear message if it is missing.

## How it works

Content and typesetting are kept separate. The `sections/*.tex` files are almost
pure semantic markup (a small, fixed vocabulary of macros); the print styling
lives in the manuscript preamble. The two stages below re-interpret that same
semantic content for the Markdown target.

1. **`build_assets.py` — the un-translatable floats become images.**
   Each `algorithm` (pseudocode) and `figure` is rendered by LaTeX itself so it
   looks identical to the PDF:
   - pseudocode is compiled in a `standalone` document that `\input`s
     [`content_preamble.tex`](content_preamble.tex) — the shared *content*
     preamble — inside a 160 mm box (the a4 text width) so `\hfill`-aligned
     comments line up exactly as in print; `\cref`s resolve to the real section
     numbers via the manuscript's `.aux` (`xr`);
   - existing figure PDFs are cropped (`pdfcrop`);
   - everything is converted to SVG (`pdftocairo -svg`) into
     `images/search_core/`, and an `assets.json` manifest records each float's
     label → image / caption / number.

2. **`latex_to_md.py` — the prose becomes primer Markdown.**
   A bespoke converter (not pandoc — the command set is small and known) walks
   the LaTeX and emits primer-flavoured Markdown, consuming `assets.json` for the
   floats and the `.aux` for cross-reference numbers. Highlights:
   - math: inline `$…$` and ` ```math ``` ` display blocks (what `build.py`
     expects); custom `\bl` bound-literals expanded; every backslash doubled so
     the mdbook markdown parser doesn't eat `\!`/`\{` before MathJax runs;
   - `\cref`/`\cite`/`\srcl…` → links; `\code`/`\emph`/`\textbf` → Markdown;
   - platypus callouts → `> [!NOTE]` / `[!WARNING]` / `:reference:` / `:log:` /
     `:tune:` box tokens (the last two were added to the top-level `build.py`);
   - assembled into one chapter: abstract + PDF link are visible in the flat
     `README.md`; the full body is wrapped in `START/STOP_SKIP_FOR_README` so it
     only appears on the mdbook website.

## Files

- `content_preamble.tex` — shared *content* preamble (`\input` by both the PDF
  build and the standalone image builder, so they can't drift).
- `build_assets.py` — stage 1: floats → SVGs + `assets.json` manifest.
- `latex_to_md.py` — stage 2: sections → `chapters/search_core.md`.
- `preview.py` — inline images into the built HTML and open it offline.
- `assets.json` — generated manifest (checked in for reference; regenerated each run).
- `_build/` — LaTeX scratch for the standalone renders (git-ignored).

The chapter and images are generated artifacts: edit the LaTeX, never the
generated Markdown. `make preview` regenerates and shows everything.
