# Sudoku propagation & conflict figures — tooling

`sudoku_search.py` is a forward-checking ("all-different elimination") Sudoku solver
that records, for every eliminated candidate, *which* peer assignment killed it and at
*what decision level*. It DFS-searches a bank of hard puzzles, snapshots every conflict
and every must-guess node, runs 1-UIP conflict analysis, scores candidates against the
two figure recipes, and renders PNG previews.

## Usage

```bash
python3 sudoku_search.py --budget 80000 --top 4 \
    --dump candidates.json --render previews
```

- `--render DIR` writes `conflict_*.png` / `guess_*.png` diagnostic previews.
- `--dump FILE` writes the top candidates (grid + analysis) as JSON for the final
  figure script.

The two **manuscript** figures (clean, title-less, no legend) are produced with:

```bash
python3 sudoku_search.py --guessfig    propagation_sudoku.pdf   # fig:sudoku-propagation
python3 sudoku_search.py --conflictfig conflict_sudoku.pdf      # fig:sudoku-conflict
```

`--colorfig DIR` renders the conflict figure(s) with titles + legend for review.

## The two figures

1. **Propagation (must-guess + cascade).** A stalled search state (root + decision
   propagation at a fixpoint, no naked single left), then a guess on a small-domain
   cell. The guessed value is struck in the same colour across its row / column /
   block (the all-different propagator), with 1–2 *secondary forced singles* showing
   the chain. Scored by `score_guess`.

2. **Conflict analysis (causal / coloured).** A cell whose candidate set was reduced
   to exactly three values by the givens (level 0), each then eliminated during search
   → empty domain. Each **eliminated value is struck in the colour of the decision that
   killed it** (the decision at the killer's decision level); arrows trace the
   propagation chains; other decisions are grey (irrelevant). The empty domain thus
   shows, *by colour*, which decisions are jointly responsible. Scored by
   `score_colored_conflict`; rendered by `draw_conflict_colored`.
   Run: `python3 sudoku_search.py --colorfig previews`.

   **Structural fact (important):** a size-3 domain is always wiped across exactly
   **two** decision levels, never three (5888 cases vs 0 in the bank). The conflict
   fires when the latest decision's propagation wave empties the *last* of the domain,
   so causes converge on **two** responsible decisions, not three. This is faithful to
   the manuscript's own abstract `fig:uipgraph`, whose learned nogood is also two
   literals (`{x4>=5, x2>=2}`) plus one off-path decision. So "two colours + several
   grey" is the honest picture, not a compromise.

   The earlier highlight-only renderer (`draw_conflict`, `score_conflict`) is kept for
   reference.

## Selected candidates (preview run)

| Figure | Puzzle | Cell | Detail |
|--------|--------|------|--------|
| Propagation | `hard17_a` | branch (2,2) dom {3,7,9}, guess 9 | 10 cross-outs, 2 forced singles, 30→27 free |
| Conflict (minimal) | `platinum` | (2,6) dom {4,5,9} | 4 decisions, 3 responsible, nogood = 2, backjump 4→1 |
| Conflict (rich) | `golden_nugget` | (5,2) dom {2,5,6} | 9 decisions, 4 responsible, nogood = 3, backjump 9→4 |

The branching heuristic is configurable (`mrv` / `fixed` / `spread`). MRV gives
realistic must-guess states; `fixed`/`spread` spread decisions across the board so a
localized conflict rests on only a subset of them (the off-path / non-chronological
cases). Conflicts are pooled across all three modes; guesses use MRV only.

## Note for the final figures

The Sudoku picture (crossing a value out of the *middle* of a domain) is the
**intuition**. CP-SAT does not keep hole-y integer domains: it propagates only the
lower/upper **bounds** of the cell variable and represents each interior elimination as
a **Boolean literal** handed to the CDCL engine — which is exactly the bound-literal
implication graph already drawn abstractly in `fig:uipgraph` (section 03). The figure
captions should make this cross-link explicit.
