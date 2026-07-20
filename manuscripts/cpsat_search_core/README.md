# How CP-SAT Reasons: The Search Core

A standalone manuscript on how Google OR-Tools' CP-SAT solver reasons under the
hood. It is written for readers who already build CP-SAT models and now want to
understand the machinery — both to model more deliberately and out of curiosity.

CP-SAT behaves at once like a constraint-programming solver, a SAT solver, and a
mixed-integer programming solver. This text follows the single loop that ties
those traditions together — propagate, branch, and learn from each dead end — and
the idea that unifies it, *lazy clause generation*: a CDCL core reasoning over
integer bounds, fed by constraint propagators and a linear relaxation. It then
covers the accelerators that make it fast in practice: lazy Boolean encoding, the
dual-simplex LP relaxation, restarts, and the parallel portfolio.

The compiled [`cpsat_search_core.pdf`](cpsat_search_core.pdf) is included in this
folder.

## Building

Requires a XeLaTeX toolchain (TeX Live) and `biber`.

```sh
make          # build cpsat_search_core.pdf
make clean    # remove LaTeX aux files
make distclean# clean + remove the PDF
```

## Layout

| Path | Contents |
| --- | --- |
| `cpsat_search_core.tex` | Main document (preamble, title page, abstract). |
| `sections/` | One file per section of the body. |
| `figures/` | Figures and the scripts that generate them. |
| `references.bib` | Bibliography. |
| `materials/` | Local reference material (talk transcripts, etc.), kept for fact-checking. **Git-ignored — not distributed.** |

## License

© Dominik Krupke. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
