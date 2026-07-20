#!/usr/bin/env python3
"""Sudoku CDCL-style solver + conflict scanner for the L10 manuscript figures.

Two figures are wanted:

  (1) PROPAGATION -- a search state where root/decision propagation has stalled
      and we must guess; then the guess (fix one value) and the same-coloured
      cross-outs it forces in the cell's row / column / block, including at least
      one *secondary* forced single (a propagation chain, not a one-hop).

  (2) CONFLICT ANALYSIS -- a cell whose candidate set was reduced to exactly
      three values {a,b,c} by the givens (level 0), then each of a,b,c killed by a
      DECISION-level assignment, emptying the domain. We want MORE decisions on the
      trail than are actually responsible, so the learned nogood is a proper subset
      (the "off-path decision stays untouched" point), and a clean 1-UIP cut so the
      backjump undoes only the last decision.

This script implements a forward-checking ("all-different elimination") Sudoku
solver that records, for every eliminated value, *which* peer assignment killed it
and at *what decision level*. It DFS-searches a set of hard puzzles, snapshots every
conflict and every must-guess node, scores them against the two figure recipes, and
prints/【dumps】 the best candidates so we can eyeball and pick.

No third-party deps for the search itself (matplotlib only for optional previews).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Optional

N = 9
CELLS = list(range(81))
ALL_VALUES = frozenset(range(1, 10))


# ---------------------------------------------------------------------------
# board geometry
# ---------------------------------------------------------------------------
def rc(cell: int) -> tuple[int, int]:
    return divmod(cell, 9)


def cell_of(r: int, c: int) -> int:
    return r * 9 + c


def _build_peers() -> list[frozenset[int]]:
    peers: list[set[int]] = [set() for _ in CELLS]
    for cell in CELLS:
        r, c = rc(cell)
        br, bc = (r // 3) * 3, (c // 3) * 3
        for k in range(9):
            peers[cell].add(cell_of(r, k))  # row
            peers[cell].add(cell_of(k, c))  # column
        for dr in range(3):
            for dc in range(3):
                peers[cell].add(cell_of(br + dr, bc + dc))  # block
        peers[cell].discard(cell)
    return [frozenset(p) for p in peers]


PEERS = _build_peers()


# ---------------------------------------------------------------------------
# solver state
# ---------------------------------------------------------------------------
@dataclass
class State:
    """A full search node. Cheap to deep-copy (81 small sets), so we copy on branch
    rather than maintaining an undo trail -- this is a 'quick solver'."""

    domains: list[set[int]]
    assigned: list[int]  # 0 = unassigned, else the value
    level: list[int]  # decision level the cell was set at (-1 unset)
    order: list[int]  # trail position when set (-1 unset)
    is_decision: list[bool]  # True if set as a branching decision
    # elim[cell][value] = (killer_cell, killer_level, killer_order):
    #   the peer assignment that removed `value` from `cell`'s domain.
    elim: list[dict[int, tuple[int, int, int]]]
    counter: int = 0  # monotone trail counter for this path

    @staticmethod
    def empty() -> "State":
        return State(
            domains=[set(range(1, 10)) for _ in CELLS],
            assigned=[0] * 81,
            level=[-1] * 81,
            order=[-1] * 81,
            is_decision=[False] * 81,
            elim=[dict() for _ in CELLS],
        )

    def copy(self) -> "State":
        return State(
            domains=[set(d) for d in self.domains],
            assigned=list(self.assigned),
            level=list(self.level),
            order=list(self.order),
            is_decision=list(self.is_decision),
            elim=[dict(e) for e in self.elim],
            counter=self.counter,
        )


class Conflict(Exception):
    """Raised by propagation when a cell domain empties. Carries the empty cell."""

    def __init__(self, cell: int):
        super().__init__(f"empty domain at cell {cell}")
        self.cell = cell


def assign(state: State, cell: int, value: int, level: int, decision: bool) -> None:
    """Assign value to cell and run forward-checking unit propagation to a fixpoint.

    Any peer that loses `value` records who killed it (cell) and when (level/order).
    Peers that collapse to a single candidate are themselves assigned (forced, same
    level) -- this is the propagation chain. Raises Conflict on an empty domain.
    """
    queue: list[tuple[int, int, bool]] = [(cell, value, decision)]
    while queue:
        c, v, is_dec = queue.pop(0)
        if state.assigned[c] == v:
            continue
        if state.assigned[c] not in (0, v):
            # trying to assign a cell two different values -> conflict at c
            raise Conflict(c)
        state.assigned[c] = v
        state.domains[c] = {v}
        state.level[c] = level
        state.order[c] = state.counter
        state.is_decision[c] = is_dec
        c_order = state.counter
        state.counter += 1
        for p in PEERS[c]:
            if state.assigned[p]:
                if state.assigned[p] == v:
                    raise Conflict(p)  # two equal values among peers
                continue
            if v in state.domains[p]:
                state.domains[p].discard(v)
                state.elim[p][v] = (c, level, c_order)
                if not state.domains[p]:
                    raise Conflict(p)
                if len(state.domains[p]) == 1:
                    (w,) = tuple(state.domains[p])
                    queue.append((p, w, False))  # forced single, same level


def load_givens(puzzle: str) -> State:
    """Apply the clues at level 0 and propagate to the root fixpoint."""
    state = State.empty()
    digits = [c for c in puzzle if c in "0123456789."]
    assert len(digits) == 81, f"expected 81 cells, got {len(digits)}"
    for i, ch in enumerate(digits):
        if ch in "0.":
            continue
        assign(state, i, int(ch), level=0, decision=False)
    return state


# ---------------------------------------------------------------------------
# 1-UIP conflict analysis
# ---------------------------------------------------------------------------
@dataclass
class Analysis:
    conflict_cell: int
    current_level: int
    root_domain: list[int]  # values still live for the conflict cell after level-0 prop
    killers: dict[
        int, tuple[int, int, int]
    ]  # value -> (killer_cell, level, order) for ALL 9 values
    uip_cell: int  # the 1-UIP literal (a cell assignment) at current level
    learned_cells: list[
        int
    ]  # cells whose assignments form the learned nogood (uip + lower levels)
    assertion_level: int  # 2nd highest level in the learned clause (backjump target)
    support_decisions: list[int]  # all decision cells reachable from the conflict
    trail_decisions: list[int]  # all decision cells on the trail at conflict time


def antecedents(state: State, cell: int) -> list[tuple[int, int, int]]:
    """Reason of a forced assignment cell=val: the killers of all *other* values."""
    val = state.assigned[cell]
    out = []
    for w, info in state.elim[cell].items():
        if w != val:
            out.append(info)
    return out


def analyze_conflict(state: State, conflict_cell: int) -> Optional[Analysis]:
    """Standard 1-UIP analysis adapted to the all-different elimination graph.

    The conflict's antecedents are the killers of every value in the empty cell.
    We resolve current-level literals in decreasing trail order until a single
    current-level literal (the 1-UIP) remains.
    """
    killers = dict(state.elim[conflict_cell])
    if len(killers) < 9:
        return None  # not every value accounted for; skip (shouldn't happen)

    levels = [lvl for (_, lvl, _) in killers.values()]
    current_level = max(levels)
    if current_level == 0:
        return None  # conflict purely from givens -> puzzle is just contradictory

    # root domain = values that survived level-0 propagation (killed at level >= 1)
    root_domain = sorted(v for v, (_, lvl, _) in killers.items() if lvl >= 1)

    # 1-UIP counter method ------------------------------------------------
    seen: set[int] = set()
    counter = 0
    learned_lower: list[int] = []  # cells at level in [1, current_level)
    uip_cell = -1

    def bump(info: tuple[int, int, int]) -> None:
        nonlocal counter
        kcell, klvl, _ = info
        if kcell in seen:
            return
        seen.add(kcell)
        if klvl == current_level:
            counter += 1
        elif klvl >= 1:
            learned_lower.append(kcell)
        # level 0 (givens) dropped

    for info in killers.values():
        bump(info)

    # walk current-level cells by decreasing trail order
    cur_cells = sorted(
        (c for c in CELLS if state.assigned[c] and state.level[c] == current_level),
        key=lambda c: -state.order[c],
    )
    for c in cur_cells:
        if c not in seen:
            continue
        if counter == 1:
            uip_cell = c
            break
        counter -= 1
        seen.discard(c)
        if state.is_decision[c]:
            uip_cell = c  # decision is the last UIP on its level
            break
        for info in antecedents(state, c):
            bump(info)
    if uip_cell == -1:
        # counter reached 1 exactly on the earliest current-level cell
        remaining = [c for c in cur_cells if c in seen]
        if not remaining:
            return None
        uip_cell = remaining[-1]

    learned_cells = [uip_cell] + sorted(set(learned_lower))
    lower_levels = [state.level[c] for c in learned_cells if c != uip_cell]
    assertion_level = max(lower_levels) if lower_levels else 0

    # full decision support (everything the conflict transitively rests on) -----
    support: set[int] = set()
    stack = [k for (k, _, _) in killers.values()]
    visited: set[int] = set()
    while stack:
        c = stack.pop()
        if c in visited:
            continue
        visited.add(c)
        if state.is_decision[c]:
            support.add(c)
        for info in antecedents(state, c):
            stack.append(info[0])
    trail_decisions = [c for c in CELLS if state.assigned[c] and state.is_decision[c]]

    return Analysis(
        conflict_cell=conflict_cell,
        current_level=current_level,
        root_domain=root_domain,
        killers=killers,
        uip_cell=uip_cell,
        learned_cells=learned_cells,
        assertion_level=assertion_level,
        support_decisions=sorted(support),
        trail_decisions=trail_decisions,
    )


# ---------------------------------------------------------------------------
# search: collect conflict snapshots and must-guess snapshots
# ---------------------------------------------------------------------------
def pick_branch_cell(state: State, mode: str = "mrv") -> int:
    """Choose an unassigned cell to branch on.

    mode="mrv"   -- smallest domain (realistic solver behaviour; tight conflicts).
    mode="fixed" -- first unassigned in row-major order (spreads decisions across
                    the board, so a localized conflict rests on only a SUBSET of
                    them -- this is what surfaces off-path / non-chronological cases).
    mode="spread"-- alternate quadrants to maximize decision spread.
    """
    unassigned = [c for c in CELLS if not state.assigned[c]]
    if not unassigned:
        return -1
    if mode == "mrv":
        return min(unassigned, key=lambda c: (len(state.domains[c]), c))
    if mode == "fixed":
        # smallest-domain among the *first few* unassigned, so branches stay legal
        # but decisions still march across the board rather than clustering
        head = sorted(unassigned)[:12]
        return min(head, key=lambda c: (len(state.domains[c]), c))
    if mode == "spread":

        def block(c):
            return (rc(c)[0] // 3, rc(c)[1] // 3)

        used = [block(c) for c in CELLS if state.assigned[c] and state.is_decision[c]]
        return min(
            unassigned, key=lambda c: (used.count(block(c)), len(state.domains[c]), c)
        )
    raise ValueError(mode)


@dataclass
class Snapshot:
    kind: str  # "conflict" or "guess"
    puzzle_id: str
    state: State
    payload: dict = field(default_factory=dict)


def search(
    puzzle: str, puzzle_id: str, node_budget: int = 60000, mode: str = "mrv"
) -> list[Snapshot]:
    """DPLL with chronological backtracking; snapshot every conflict and the
    must-guess nodes. `mode` selects the branching heuristic (see pick_branch_cell)."""
    snaps: list[Snapshot] = []
    root = load_givens(puzzle)

    # record the very first must-guess state (root propagation done)
    if not is_solved(root):
        snaps.append(Snapshot("guess", puzzle_id, root.copy(), {"depth": 0}))

    nodes = 0

    def dfs(state: State, level: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            return False
        if is_solved(state):
            return True
        cell = pick_branch_cell(state, mode)
        if cell == -1:
            return False
        # snapshot must-guess states that are readable: few free cells, propagation
        # already at a fixpoint (no naked single waiting). Captured by free-count, not
        # depth -- hard puzzles still have ~50 free cells at shallow depth.
        free_count = sum(1 for c in CELLS if not state.assigned[c])
        if free_count <= 34 and len(snaps) < 12000:
            snaps.append(
                Snapshot(
                    "guess",
                    puzzle_id,
                    state.copy(),
                    {"depth": level, "branch_cell": cell},
                )
            )
        for value in sorted(state.domains[cell]):
            child = state.copy()
            try:
                assign(child, cell, value, level=level + 1, decision=True)
            except Conflict as cf:
                ana = analyze_conflict(child, cf.cell)
                if ana is not None:
                    snaps.append(
                        Snapshot(
                            "conflict",
                            puzzle_id,
                            child,
                            {"analysis": ana, "depth": level + 1},
                        )
                    )
                continue
            if dfs(child, level + 1):
                return True
        return False

    dfs(root, 0)
    return snaps


def is_solved(state: State) -> bool:
    return all(state.assigned[c] for c in CELLS)


# ---------------------------------------------------------------------------
# scoring the candidates against the two figure recipes
# ---------------------------------------------------------------------------
def score_conflict(snap: Snapshot) -> Optional[dict]:
    ana: Analysis = snap.payload["analysis"]
    if len(ana.root_domain) != 3:
        return None  # we specifically want a size-3 wipeout
    n_trail = len(ana.trail_decisions)
    n_support = len(ana.support_decisions)
    irrelevant = n_trail - n_support
    learned = len(ana.learned_cells)
    backjump = ana.current_level - ana.assertion_level  # >1 = non-chronological
    narrowing = n_support - learned  # decisions the nogood drops
    # The CDCL story we want to draw: a size-3 wipeout traced to a handful of
    # responsible decisions, of which the 1-UIP nogood keeps only a SUBSET
    # (narrowing >= 1), and a real backjump that undoes several decisions at once
    # without revisiting them (backjump >= 2). Keep the board drawable: support <= 4.
    if learned < 2 or learned > 3:
        return None
    if not (3 <= n_support <= 4):  # backtrace to ~three responsible decisions
        return None
    if narrowing < 1 and irrelevant < 1:
        return None  # nogood must be a strict subset of the trail
    if backjump < 2:
        return None  # want a genuine non-chronological jump
    score = (
        100
        + 12 * min(narrowing, 3)  # reward the nogood narrowing the culprits
        + 10 * min(irrelevant, 3)  # reward provably-irrelevant decisions
        + 9 * min(backjump, 5)  # reward a big backjump
        + 8 * max(0, 4 - learned)  # reward a small learned nogood
        - 6 * max(0, n_support - 3)  # prefer a tight support (<=3)
        - 3 * max(0, n_trail - 9)  # mild penalty for very deep trails
    )
    return {
        "score": score,
        "conflict_cell": rc(ana.conflict_cell),
        "root_domain": ana.root_domain,
        "trail_decisions": n_trail,
        "support_decisions": n_support,
        "irrelevant_decisions": irrelevant,
        "learned_size": learned,
        "current_level": ana.current_level,
        "assertion_level": ana.assertion_level,
        "backjump": backjump,
    }


def score_guess(snap: Snapshot) -> Optional[dict]:
    state = snap.state
    free = [c for c in CELLS if not state.assigned[c]]
    if not (10 <= len(free) <= 30):
        return None  # readable but non-trivial; bigger domains => visible cross-outs
    # must-guess: no unassigned singleton remains (propagation is at a fixpoint)
    if any(len(state.domains[c]) == 1 and not state.assigned[c] for c in free):
        return None
    # prefer a small-domain branch cell (a real 2- or 3-way guess)
    cand_cells = [c for c in free if 2 <= len(state.domains[c]) <= 3]
    if not cand_cells:
        return None
    best = None
    for branch in cand_cells:
        for v in sorted(state.domains[branch]):
            child = state.copy()
            try:
                assign(child, branch, v, level=1, decision=True)
            except Conflict:
                continue
            # direct cross-outs: peers of the guessed cell that lose value v.
            # Split into the two visuals we care about:
            #   struck_open  -- peer keeps >=2 candidates: a VISIBLE pencil-mark strike
            #   collapsed    -- peer drops to a single value: a forced-single chain step
            direct = [
                p
                for p in PEERS[branch]
                if not state.assigned[p] and v in state.domains[p]
            ]
            struck_open = [p for p in direct if len(state.domains[p]) >= 3]
            secondary = [c for c in free if c != branch and child.assigned[c]]
            free_after = sum(1 for c in free if not child.assigned[c])
            # The figure-1 ideal: enough total same-colour strikes, at least a couple
            # that leave the cell visibly open, plus 1-2 forced singles for the chain.
            if (
                len(direct) < 3
                or len(struck_open) < 1
                or not (1 <= len(secondary) <= 3)
            ):
                continue
            if free_after < 6:
                continue
            score = (
                40
                + 9 * len(struck_open)  # the headline visual (open strikes)
                + 3 * (len(direct) - len(struck_open))  # collapsed strikes
                + 6 * len(secondary)  # the chain
                - abs(len(free) - 20)  # readability sweet spot
            )
            cand = {
                "score": score,
                "free_cells": len(free),
                "branch_cell": rc(branch),
                "branch_domain": sorted(state.domains[branch]),
                "guess_value": v,
                "direct_crossouts": [rc(p) for p in direct],
                "struck_open": [rc(p) for p in struck_open],
                "secondary_singles": [rc(c) for c in secondary],
                "free_after_guess": free_after,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------
def render_ascii(state: State, highlight: Optional[set[int]] = None) -> str:
    highlight = highlight or set()
    lines = []
    for r in range(9):
        if r % 3 == 0:
            lines.append("+-------+-------+-------+")
        row = []
        for c in range(9):
            cell = cell_of(r, c)
            if c % 3 == 0:
                row.append("|")
            if state.assigned[cell]:
                ch = str(state.assigned[cell])
                ch = f"({ch})" if cell in highlight else f" {ch} "
                row.append(ch.strip().center(1) if len(ch) == 1 else ch[1])
                row.append("")
            else:
                row.append(".")
            row.append(" ")
        row.append("|")
        lines.append("".join(row).replace("  ", " "))
    lines.append("+-------+-------+-------+")
    return "\n".join(lines)


def render_pencil(state: State) -> str:
    """Compact candidate grid: assigned cells show the digit, free cells show their
    candidate set (sorted)."""
    lines = []
    for r in range(9):
        if r % 3 == 0:
            lines.append("-" * 64)
        cells = []
        for c in range(9):
            cell = cell_of(r, c)
            sep = "| " if c % 3 == 0 else " "
            if state.assigned[cell]:
                cells.append(sep + f"[{state.assigned[cell]}]".ljust(6))
            else:
                cand = "".join(str(v) for v in sorted(state.domains[cell]))
                cells.append(sep + cand.ljust(6))
        lines.append("".join(cells) + "|")
    lines.append("-" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# matplotlib previews (diagnostic, not the final figure)
# ---------------------------------------------------------------------------
def _grid_axes(ax, title):
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for k in range(10):
        lw = 2.2 if k % 3 == 0 else 0.6
        ax.plot([k, k], [0, 9], color="black", lw=lw)
        ax.plot([0, 9], [k, k], color="black", lw=lw)
    if title:
        ax.set_title(title, fontsize=10)


def _pencil_pos(x, y, v):
    """Position of candidate value v in a cell's 3x3 pencil-mark grid."""
    return (x + 0.20 + 0.30 * ((v - 1) % 3), y + 0.22 + 0.30 * ((v - 1) // 3))


def _pencil_mark(
    ax,
    x,
    y,
    v,
    color="#888888",
    fontsize=12,
    bold=False,
    strike=False,
    strike_color=None,
):
    """Draw a single pencil-mark candidate, optionally struck through."""
    import matplotlib.pyplot as plt  # noqa: F401 (ax already bound to a figure)

    gx, gy = _pencil_pos(x, y, v)
    ax.text(
        gx,
        gy,
        str(v),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight="bold" if bold else "normal",
    )
    if strike:
        ax.plot(
            [gx - 0.13, gx + 0.13],
            [gy + 0.08, gy - 0.08],
            color=strike_color or color,
            lw=1.3,
        )


def draw_conflict(snap: "Snapshot", ana: Analysis, path: str) -> None:
    import matplotlib.pyplot as plt

    state = snap.state
    learned = set(ana.learned_cells)
    killer_cells = {info[0] for v, info in ana.killers.items() if v in ana.root_domain}
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _grid_axes(
        ax,
        f"{snap.puzzle_id}: conflict @ {rc(ana.conflict_cell)}  "
        f"dom={ana.root_domain}  trail={len(ana.trail_decisions)} "
        f"resp={len(ana.support_decisions)} learn={len(ana.learned_cells)} "
        f"bj {ana.current_level}->{ana.assertion_level}",
    )
    for cell in CELLS:
        r, c = rc(cell)
        x, y = c, r
        if cell == ana.conflict_cell:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color="#f3b0b0"))
        elif cell in learned:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color="#bfe3bf"))
        elif cell in killer_cells:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color="#ffe0a3"))
        if cell == ana.conflict_cell:
            # three struck candidates + empty-domain marker
            for i, v in enumerate(ana.root_domain):
                ax.text(
                    x + 0.25 + 0.25 * i,
                    y + 0.32,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#b00000",
                )
                ax.plot(
                    [x + 0.12 + 0.25 * i, x + 0.38 + 0.25 * i],
                    [y + 0.40, y + 0.24],
                    color="#b00000",
                    lw=1.4,
                )
            ax.text(
                x + 0.5,
                y + 0.72,
                "⊥",
                ha="center",
                va="center",
                fontsize=14,
                color="#b00000",
                fontweight="bold",
            )
            continue
        if state.assigned[cell]:
            val = state.assigned[cell]
            if state.is_decision[cell]:
                color, weight = "#1f4fd6", "bold"
                ax.text(
                    x + 0.78,
                    y + 0.22,
                    f"d{state.level[cell]}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#1f4fd6",
                )
            elif state.level[cell] == 0:
                color, weight = "black", "normal"
            else:
                color, weight = "#888888", "normal"
            ax.text(
                x + 0.5,
                y + 0.55,
                str(val),
                ha="center",
                va="center",
                fontsize=15,
                color=color,
                fontweight=weight,
            )
            if cell in learned:
                ax.add_patch(
                    plt.Circle(
                        (x + 0.5, y + 0.5), 0.40, fill=False, color="#157a15", lw=2.0
                    )
                )
        else:
            cand = sorted(state.domains[cell])
            ax.text(
                x + 0.5,
                y + 0.5,
                "".join(map(str, cand)),
                ha="center",
                va="center",
                fontsize=6.5,
                color="#666666",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _decision_at_level(state: State, lvl: int) -> int:
    for c in CELLS:
        if state.assigned[c] and state.is_decision[c] and state.level[c] == lvl:
            return c
    return -1


def _replay_before(state: State, puzzle: str, trig_level: int) -> State:
    """Reconstruct the propagation fixpoint just before the decision at trig_level
    was made: re-apply the givens and every earlier decision (re-propagating)."""
    decs = sorted(
        (
            c
            for c in CELLS
            if state.assigned[c]
            and state.is_decision[c]
            and 1 <= state.level[c] < trig_level
        ),
        key=lambda c: state.level[c],
    )
    pre = load_givens(puzzle)
    for d in decs:
        assign(pre, d, state.assigned[d], level=state.level[d], decision=True)
    return pre


def _path_to_decision(state: State, killer: int, dec: int) -> list[int]:
    """Shortest antecedent path dec -> ... -> killer (cause to effect)."""
    if killer == dec:
        return [killer]
    from collections import deque

    prev = {killer: None}
    q = deque([killer])
    while q:
        c = q.popleft()
        if c == dec:
            break
        if state.is_decision[c] or state.level[c] == 0:
            continue
        for kc, lvl, _ in antecedents(state, c):
            if kc not in prev:
                prev[kc] = c
                q.append(kc)
    if dec not in prev:
        return [killer]
    path = []
    c = dec
    while c is not None:
        path.append(c)
        c = prev[c]
    return path  # dec ... killer (cause -> effect order)


def score_colored_conflict(snap: "Snapshot") -> Optional[dict]:
    """Score a conflict for the COLOURED causal figure: size-3 wipeout, exactly two
    distinct responsible decisions (two colours), short drawable chains, and several
    irrelevant decisions on the trail."""
    ana: Analysis = snap.payload["analysis"]
    if len(ana.root_domain) != 3:
        return None
    state = snap.state
    resp = {v: _decision_at_level(state, ana.killers[v][1]) for v in ana.root_domain}
    if any(d < 0 for d in resp.values()):
        return None
    ncolors = len(set(resp.values()))
    if ncolors != 2:
        return None
    n_trail = len(ana.trail_decisions)
    if not (5 <= n_trail <= 9):
        return None
    chain_edges = 0
    for v in ana.root_domain:
        chain_edges += max(
            0, len(_path_to_decision(state, ana.killers[v][0], resp[v])) - 1
        )
    irrelevant = n_trail - ncolors
    score = 100 + 8 * min(irrelevant, 5) - 9 * chain_edges - 2 * abs(n_trail - 6)
    return {
        "score": score,
        "conflict_cell": rc(ana.conflict_cell),
        "root_domain": ana.root_domain,
        "trail": n_trail,
        "colors": ncolors,
        "chain_hops": chain_edges,
        "irrelevant": irrelevant,
    }


def draw_conflict_colored(
    snap: "Snapshot", ana: Analysis, path: str, clean: bool = False
) -> None:
    """Causal figure, in the SAME visual language as the propagation figure:
      * a DECISION (guess) is a big bold digit in a solid-bordered cell, tagged "D";
      * a PROPAGATION is a cell whose domain is shown reduced (the upstream value
        struck, the survivor in bold) in a dashed-bordered cell;
      * colour = which decision is ultimately responsible;
      * the conflict cell is the final propagation whose whole domain is struck out
        (no survivor) -> empty domain, marked with the conflict symbol.
    Tracing the colours back shows that only a SUBSET of the decisions matters."""
    import matplotlib.pyplot as plt

    state = snap.state
    conflict = ana.conflict_cell
    # responsible decision per eliminated value = the decision at the killer's level
    resp = {v: _decision_at_level(state, ana.killers[v][1]) for v in ana.root_domain}
    resp_decs = [d for d in dict.fromkeys(resp.values()) if d >= 0]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    dec_color = {d: palette[i % len(palette)] for i, d in enumerate(resp_decs)}

    # responsible chains: a decision may not strike the conflict cell directly but
    # through a forced single (e.g. (0,2)=5 forces (2,0)=6, and that 6 empties the
    # conflict cell).  Colour those intermediate cells -- the upstream value struck
    # and the survivor that carries the impact onward -- in the decision's colour.
    chain_color: dict[int, str] = {}  # cell -> colour (survivor + chain marks)
    chain_kill: dict[int, set] = {}  # cell -> upstream values struck on the chain
    for v in ana.root_domain:
        dec, killer = resp[v], ana.killers[v][0]
        if dec < 0:
            continue
        col = dec_color[dec]
        chain = _path_to_decision(state, killer, dec)  # [dec, ..., killer]
        for a, b in zip(chain, chain[1:]):  # a forces b by losing assigned[a]
            if b == conflict:
                continue
            chain_color[b] = col
            chain_kill.setdefault(b, set()).add(state.assigned[a])

    # the decision that triggers the conflict ("deciding the 5")
    trig_level = ana.current_level
    trigger = _decision_at_level(state, trig_level)

    # the candidate set every open cell starts from = the givens fixpoint.  Pencil
    # marks are drawn from this; propagation then strikes the ones a guess removed.
    init = load_givens(PUZZLES[snap.puzzle_id])

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.4))

    def fill(ax, x, y, col, a):
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=col, alpha=a, lw=0))

    def border(ax, x, y, col, style="solid", lw=2.2):
        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, ec=col, lw=lw, ls=style))

    def shown_as_number(st, cell):
        """A cell printed as a big digit: a given or a branching decision (not a
        propagated/forced single, which stays a pencil grid)."""
        return st.assigned[cell] and (st.level[cell] == 0 or st.is_decision[cell])

    def glyph(st, cell):
        """Where an arrow out of `cell` starts: a digit's centre, or, for a forced
        single shown as pencil marks, the surviving mark."""
        r, c = rc(cell)
        if shown_as_number(st, cell):
            return (c + 0.5, r + 0.55)
        if st.assigned[cell]:
            return _pencil_pos(c, r, st.assigned[cell])
        return (c + 0.5, r + 0.5)

    def render(ax, st, colored, title):
        """1. givens as fixed numbers, free cells as pencil-mark domains.
        2. the decisions (big digit + box + D).
        3. propagation: candidates a peer removed are struck; `colored` lights up the
           responsible decisions and the (now empty) conflict cell, else all grey."""
        _grid_axes(ax, "" if clean else title)
        for cell in CELLS:
            r, c = rc(cell)
            x, y = c, r
            if st.assigned[cell] and st.level[cell] == 0:  # a given
                ax.text(
                    x + 0.5,
                    y + 0.55,
                    str(st.assigned[cell]),
                    ha="center",
                    va="center",
                    fontsize=18,
                    color="black",
                )
                continue
            if st.assigned[cell] and st.is_decision[cell]:  # a decision (guess)
                if colored and cell in dec_color:  # a responsible guess
                    col, lw = dec_color[cell], 2.2
                    fill(ax, x, y, col, 0.16)
                else:  # an off-path guess
                    col, lw = "#5f646b", 1.3
                    fill(ax, x, y, "#aab0b6", 0.28)  # light grey background
                border(ax, x, y, col, "solid", lw)
                ax.text(
                    x + 0.5,
                    y + 0.55,
                    str(st.assigned[cell]),
                    ha="center",
                    va="center",
                    fontsize=15,
                    color=col,
                    fontweight="bold",
                )
                ax.text(
                    x + 0.15,
                    y + 0.16,
                    "D",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=col,
                    fontweight="bold",
                )
                continue
            # otherwise a pencil-mark cell: free, or a forced single
            is_conf = cell == conflict
            on_chain = colored and cell in chain_color  # carries a guess onward
            if colored and is_conf:  # the empty domain
                fill(ax, x, y, "#b00000", 0.10)
                border(ax, x, y, "#b00000", "solid", 2.4)
            elif on_chain:
                fill(ax, x, y, chain_color[cell], 0.08)
            for v in sorted(init.domains[cell]):
                info = st.elim[cell].get(v)
                killed = info is not None and info[1] >= 1 and v not in st.domains[cell]
                if not killed:
                    if on_chain and st.assigned[cell] == v:  # the survivor carrying on
                        _pencil_mark(
                            ax, x, y, v, color=chain_color[cell], fontsize=14, bold=True
                        )
                    else:
                        _pencil_mark(ax, x, y, v)  # still live (grey)
                elif colored and is_conf:  # who emptied it
                    col = dec_color.get(resp.get(v, -1), "#b00000")
                    _pencil_mark(
                        ax,
                        x,
                        y,
                        v,
                        color=col,
                        fontsize=14,
                        bold=True,
                        strike=True,
                        strike_color=col,
                    )
                elif on_chain and v in chain_kill[cell]:  # struck on the chain
                    col = chain_color[cell]
                    _pencil_mark(
                        ax,
                        x,
                        y,
                        v,
                        color=col,
                        fontsize=12,
                        bold=True,
                        strike=True,
                        strike_color=col,
                    )
                else:  # struck by propagation
                    _pencil_mark(
                        ax,
                        x,
                        y,
                        v,
                        color="#aab0b6",
                        fontsize=12,
                        strike=True,
                        strike_color="#aab0b6",
                    )
            if colored and is_conf:
                ax.text(
                    x + 0.86,
                    y + 0.14,
                    "⊥",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#b00000",
                    fontweight="bold",
                )

    def render_arrows(ax, st, colored):
        """One arrow per struck pencil mark: from the assignment that removed it to
        the mark.  Light grey for ordinary propagation; coloured for the eliminations
        that empty the conflict cell."""
        for cell in CELLS:
            if shown_as_number(st, cell):
                continue
            r, c = rc(cell)
            for v in sorted(init.domains[cell]):
                info = st.elim[cell].get(v)
                if info is None or info[1] < 1 or v in st.domains[cell]:
                    continue
                src = glyph(st, info[0])
                dst = _pencil_pos(c, r, v)
                if colored and cell == conflict:
                    col, a, lw = dec_color.get(resp.get(v, -1), "#b00000"), 0.85, 2.2
                elif colored and cell in chain_color and v in chain_kill[cell]:
                    col, a, lw = chain_color[cell], 0.85, 2.2
                else:
                    col, a, lw = "#c2c7cc", 0.65, 1.3
                ax.annotate(
                    "",
                    xy=dst,
                    xytext=src,
                    arrowprops=dict(
                        arrowstyle="->",
                        color=col,
                        lw=lw,
                        alpha=a,
                        mutation_scale=8,
                        shrinkA=3,
                        shrinkB=2,
                        connectionstyle="arc3,rad=0.12",
                    ),
                )

    # LEFT: every guess but the trigger, placed and propagated -- all neutral grey,
    # nothing wrong yet (the conflict cell is merely reduced, no red).
    pre = _replay_before(state, PUZZLES[snap.puzzle_id], trig_level)
    render(
        axes[0],
        pre,
        colored=False,
        title=f"{snap.puzzle_id}: every guess but {rc(trigger)} placed and propagated",
    )
    render_arrows(axes[0], pre, colored=False)

    # RIGHT: add the trigger guess; its propagation strikes the rest of the conflict
    # cell's domain -> empty -> conflict.  The responsible guesses light up.
    render(
        axes[1],
        state,
        colored=True,
        title=f"deciding {rc(trigger)}={state.assigned[trigger]} → conflict @ "
        f"{rc(conflict)}: only {len(resp_decs)} of "
        f"{len(ana.trail_decisions)} decisions involved",
    )
    render_arrows(axes[1], state, colored=True)

    # legend: the encoding, then which decisions are responsible
    if not clean:
        fig.text(
            0.02,
            0.085,
            "black = given      boxed + D = decision (guess)      "
            "struck pencil mark = candidate a peer removed "
            "(arrow = by which assignment)",
            fontsize=8.5,
            color="#444444",
        )
        parts = []
        for d in resp_decs:
            kills = [str(v) for v in ana.root_domain if resp[v] == d]
            parts.append(
                (
                    dec_color[d],
                    f"decision {rc(d)}={state.assigned[d]} → empties "
                    f"{{{','.join(kills)}}} of {rc(conflict)}",
                )
            )
        for i, (col, txt) in enumerate(parts):
            fig.text(
                0.02, 0.052 - 0.024 * i, txt, color=col, fontsize=8.5, fontweight="bold"
            )
        fig.text(
            0.58,
            0.052,
            f"{len(ana.trail_decisions) - len(resp_decs)} other "
            f"decisions never touch {rc(conflict)}",
            color="#999999",
            fontsize=8.5,
        )

    fig.tight_layout(rect=[0, 0.11, 1, 0.95] if not clean else [0, 0, 1, 1])
    fig.savefig(path, dpi=130, bbox_inches="tight" if clean else None)
    plt.close(fig)


def draw_guess(snap: "Snapshot", sc: dict, path: str, clean: bool = False) -> None:
    import matplotlib.pyplot as plt

    state = snap.state
    branch = cell_of(*sc["branch_cell"])
    direct = {cell_of(*p) for p in sc["direct_crossouts"]}
    secondary = {cell_of(*p) for p in sc["secondary_singles"]}
    gv = sc["guess_value"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4))
    # left: before the guess (must-guess state, pencil marks)
    _grid_axes(
        axes[0],
        ""
        if clean
        else f"{snap.puzzle_id}: stalled, must guess  "
        f"(branch {sc['branch_cell']} dom={sc['branch_domain']})",
    )
    # right: after guessing branch=gv and propagating
    child = state.copy()
    assign(child, branch, gv, level=1, decision=True)
    _grid_axes(
        axes[1],
        ""
        if clean
        else f"guess {sc['branch_cell']}={gv} -> {len(direct)} cross-outs, "
        f"{len(sc['secondary_singles'])} forced singles",
    )

    def pencil(ax, x, y, cands, strike=(), survivor=None):
        """Render a 3x3 candidate grid; strike the values in `strike` (red) and, if a
        `survivor` is given, draw it bold green (the value the domain was forced to)."""
        strike = set(strike)
        for v in cands:
            gx = x + 0.20 + 0.30 * ((v - 1) % 3)
            gy = y + 0.22 + 0.30 * ((v - 1) // 3)
            if v == survivor:
                ax.text(
                    gx,
                    gy,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="#157a15",
                    fontweight="bold",
                )
            elif v in strike:
                ax.text(
                    gx,
                    gy,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#b00000",
                )
                ax.plot(
                    [gx - 0.13, gx + 0.13],
                    [gy + 0.08, gy - 0.08],
                    color="#b00000",
                    lw=1.3,
                )
            else:
                ax.text(
                    gx,
                    gy,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#888888",
                )

    for ax, st, after in ((axes[0], state, False), (axes[1], child, True)):
        for cell in CELLS:
            r, c = rc(cell)
            x, y = c, r
            if cell == branch:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color="#cfe0ff"))
            elif after and cell in secondary:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color="#d9f0d9"))
            # a newly forced single: show HOW its domain was cut to one -- the parent
            # candidates with the eliminated ones struck and the survivor bold green
            if after and cell in secondary:
                survivor = child.assigned[cell]
                parent = sorted(state.domains[cell])
                pencil(
                    ax,
                    x,
                    y,
                    parent,
                    strike=[v for v in parent if v != survivor],
                    survivor=survivor,
                )
            elif st.assigned[cell]:
                val = st.assigned[cell]
                color = "black" if st.level[cell] == 0 else "#444444"
                if cell == branch:
                    color = "#1f4fd6"
                ax.text(
                    x + 0.5,
                    y + 0.55,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=18 if st.level[cell] == 0 else 15,
                    color=color,
                    fontweight="bold" if cell == branch else "normal",
                )
            else:
                # on the right board, render the PARENT candidates and strike every
                # value the guess *or* its forced-single cascade eliminated, so no
                # candidate ever silently disappears between the two boards.
                if after and not child.assigned[cell]:
                    parent = sorted(state.domains[cell])
                    killed = [v for v in parent if v not in child.domains[cell]]
                    pencil(ax, x, y, parent, strike=killed)
                else:
                    pencil(ax, x, y, sorted(st.domains[cell]))

    # thin propagation arrows on the right board: from each killer's value glyph to
    # the candidate it struck.  Blue (the guess) -> every crossed-out 9; green (a
    # forced single) -> the value it in turn strikes (the 8 -> 9 chain).
    def _glyph(cell, v):
        r, c = rc(cell)
        if cell == branch:
            return (c + 0.5, r + 0.55)
        return (c + 0.20 + 0.30 * ((v - 1) % 3), r + 0.22 + 0.30 * ((v - 1) // 3))

    def _arrow(ax, killer, cell, v):
        kv = child.assigned[killer]
        x0, y0 = _glyph(killer, kv)
        x1, y1 = _glyph(cell, v)
        col = "#1f4fd6" if killer == branch else "#157a15"
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=col,
                lw=1.4,
                alpha=0.7,
                mutation_scale=8,
                shrinkA=4,
                shrinkB=4,
                connectionstyle="arc3,rad=0.14",
            ),
        )

    axr = axes[1]
    for cell in CELLS:
        if child.assigned[cell] or cell in secondary:
            continue
        for v in state.domains[cell]:
            if v not in child.domains[cell] and v in child.elim[cell]:
                _arrow(axr, child.elim[cell][v][0], cell, v)
    for cell in secondary:
        survivor = child.assigned[cell]
        for v in state.domains[cell]:
            if v != survivor and v in child.elim[cell]:
                _arrow(axr, child.elim[cell][v][0], cell, v)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight" if clean else None)
    plt.close(fig)


# ---------------------------------------------------------------------------
# puzzle bank (well-known hard, unique-solution Sudokus)
# ---------------------------------------------------------------------------
PUZZLES = {
    "ai_escargot": "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
    "inkala2012": "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
    "inkala2010": "005300000800000020070010500400005300010070006003200080060500009004000030000009700",
    "platinum": "000000012000035000000600070700000300000400800100000000000120000080000040050000600",
    "golden_nugget": "000000039000001005003050800008090006070002000100400000009080050020000600400700000",
    "hard17_a": "000000010400000000020000000000050407008000300001090000300400200050100000000806000",
    "hard17_b": "000000010040000000000000000700050407008000300001090000300400200050100000000806000",
    "easter_monster": "100000002090400050006000700050903000000070000000850040700000600030009080002000001",
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60000, help="node budget per puzzle")
    ap.add_argument("--top", type=int, default=5, help="how many candidates to show")
    ap.add_argument(
        "--dump", type=str, default=None, help="write best candidates to JSON"
    )
    ap.add_argument(
        "--render", type=str, default=None, help="render top candidates as PNG into DIR"
    )
    ap.add_argument(
        "--colorfig",
        type=str,
        default=None,
        help="render the best COLOURED causal conflict figures into DIR",
    )
    ap.add_argument(
        "--guessfig",
        type=str,
        default=None,
        help="render the single best propagation figure (clean, no titles) to this PDF path",
    )
    ap.add_argument(
        "--conflictfig",
        type=str,
        default=None,
        help="render the single best conflict figure (clean, no titles) to this PDF path",
    )
    args = ap.parse_args()

    if args.conflictfig:
        pool = []
        for pid, puzzle in PUZZLES.items():
            for mode in ("mrv", "fixed", "spread"):
                try:
                    snaps = search(puzzle, pid, node_budget=args.budget, mode=mode)
                except Conflict:
                    break
                seen = set()
                for s in snaps:
                    if s.kind != "conflict":
                        continue
                    sc = score_colored_conflict(s)
                    if not sc:
                        continue
                    a = s.payload["analysis"]
                    sig = (
                        pid,
                        a.conflict_cell,
                        tuple(sorted(a.killers[v][1] for v in a.root_domain)),
                    )
                    if sig in seen:
                        continue
                    seen.add(sig)
                    pool.append((sc["score"], pid, s, sc))
        pool.sort(key=lambda x: -x[0])
        _, pid, s, sc = pool[0]
        draw_conflict_colored(s, s.payload["analysis"], args.conflictfig, clean=True)
        print(
            f"rendered clean conflict figure -> {args.conflictfig} "
            f"({pid}, trail={sc['trail']}, irrelevant={sc['irrelevant']})",
            file=sys.stderr,
        )
        return

    if args.colorfig:
        import os

        os.makedirs(args.colorfig, exist_ok=True)
        pool = []
        for pid, puzzle in PUZZLES.items():
            for mode in ("mrv", "fixed", "spread"):
                try:
                    snaps = search(puzzle, pid, node_budget=args.budget, mode=mode)
                except Conflict:
                    break
                seen = set()
                for s in snaps:
                    if s.kind != "conflict":
                        continue
                    sc = score_colored_conflict(s)
                    if not sc:
                        continue
                    a = s.payload["analysis"]
                    sig = (
                        pid,
                        a.conflict_cell,
                        tuple(sorted(a.killers[v][1] for v in a.root_domain)),
                    )
                    if sig in seen:
                        continue
                    seen.add(sig)
                    pool.append((sc["score"], pid, s, sc))
        pool.sort(key=lambda x: -x[0])
        for i, (score, pid, s, sc) in enumerate(pool[: args.top]):
            a = s.payload["analysis"]
            p = os.path.join(args.colorfig, f"colorfig_{i}_{pid}_{a.conflict_cell}.png")
            draw_conflict_colored(s, a, p)
            print(
                f"rendered {p}  trail={sc['trail']} hops={sc['chain_hops']} "
                f"irrelevant={sc['irrelevant']}",
                file=sys.stderr,
            )
        return

    all_conflicts: list[tuple[dict, Snapshot]] = []
    all_guesses: list[tuple[dict, Snapshot]] = []

    # MRV gives realistic must-guess states; "fixed"/"spread" spread decisions and
    # surface conflicts with off-path / non-chronological structure. Pool all.
    for pid, puzzle in PUZZLES.items():
        n_c = n_g = 0
        for mode in ("mrv", "fixed", "spread"):
            try:
                snaps = search(puzzle, pid, node_budget=args.budget, mode=mode)
            except Conflict:
                if mode == "mrv":
                    print(
                        f"[{pid}] givens already contradictory -- skipped",
                        file=sys.stderr,
                    )
                break
            for s in snaps:
                if s.kind == "conflict":
                    sc = score_conflict(s)
                    if sc:
                        all_conflicts.append((sc, s))
                        n_c += 1
                elif mode == "mrv":  # only trust MRV for realistic guesses
                    sc = score_guess(s)
                    if sc:
                        all_guesses.append((sc, s))
                        n_g += 1
        print(f"[{pid}] -> {n_c} conflict / {n_g} guess candidates", file=sys.stderr)

    all_conflicts.sort(key=lambda x: -x[0]["score"])
    all_guesses.sort(key=lambda x: -x[0]["score"])

    def dedupe(items, sig):
        seen, out = set(), []
        for it in items:
            k = sig(it)
            if k in seen:
                continue
            seen.add(k)
            out.append(it)
        return out

    all_conflicts = dedupe(
        all_conflicts,
        lambda x: (
            x[1].puzzle_id,
            x[1].payload["analysis"].conflict_cell,
            tuple(sorted(x[1].payload["analysis"].learned_cells)),
            x[0]["trail_decisions"],
        ),
    )
    all_guesses = dedupe(
        all_guesses,
        lambda x: (x[1].puzzle_id, x[0]["branch_cell"], x[0]["free_cells"]),
    )

    print("\n" + "=" * 72)
    print("BEST CONFLICT-ANALYSIS CANDIDATES (size-3 domain wiped out)")
    print("=" * 72)
    for sc, snap in all_conflicts[: args.top]:
        ana: Analysis = snap.payload["analysis"]
        print(
            f"\n[{snap.puzzle_id}] score={sc['score']}  conflict cell={sc['conflict_cell']} "
            f"root domain={sc['root_domain']}"
        )
        print(
            f"   decisions on trail={sc['trail_decisions']}  responsible={sc['support_decisions']}  "
            f"irrelevant={sc['irrelevant_decisions']}"
        )
        print(
            f"   learned nogood size={sc['learned_size']}  current level={sc['current_level']}  "
            f"backjump to level {sc['assertion_level']} (jump {sc['backjump']})"
        )
        print(f"   learned cells: {[rc(c) for c in ana.learned_cells]}")
        print(
            render_ascii(
                snap.state, highlight=set(ana.learned_cells) | {ana.conflict_cell}
            )
        )

    print("\n" + "=" * 72)
    print("BEST PROPAGATION (MUST-GUESS) CANDIDATES")
    print("=" * 72)
    for sc, snap in all_guesses[: args.top]:
        print(
            f"\n[{snap.puzzle_id}] score={sc['score']}  free cells={sc['free_cells']}  "
            f"branch={sc['branch_cell']} domain={sc['branch_domain']} -> guess {sc['guess_value']}"
        )
        print(
            f"   direct cross-outs={len(sc['direct_crossouts'])} {sc['direct_crossouts']}"
        )
        print(
            f"   secondary singles={len(sc['secondary_singles'])} {sc['secondary_singles']}  "
            f"free after guess={sc['free_after_guess']}"
        )
        print(render_pencil(snap.state))

    if args.render:
        import os

        os.makedirs(args.render, exist_ok=True)
        for i, (sc, snap) in enumerate(all_conflicts[: args.top]):
            ana = snap.payload["analysis"]
            p = os.path.join(args.render, f"conflict_{i}_{snap.puzzle_id}.png")
            draw_conflict(snap, ana, p)
            print(f"rendered {p}", file=sys.stderr)
        for i, (sc, snap) in enumerate(all_guesses[: args.top]):
            p = os.path.join(args.render, f"guess_{i}_{snap.puzzle_id}.png")
            draw_guess(snap, sc, p)
            print(f"rendered {p}", file=sys.stderr)

    if args.guessfig:
        sc, snap = all_guesses[0]
        draw_guess(snap, sc, args.guessfig, clean=True)
        print(f"rendered clean propagation figure -> {args.guessfig}", file=sys.stderr)

    if args.dump:
        out = {
            "conflicts": [
                {
                    "puzzle": snap.puzzle_id,
                    "grid": [snap.state.assigned[c] for c in CELLS],
                    "score": sc,
                    "analysis": {
                        "conflict_cell": snap.payload["analysis"].conflict_cell,
                        "root_domain": snap.payload["analysis"].root_domain,
                        "learned_cells": snap.payload["analysis"].learned_cells,
                        "support_decisions": snap.payload["analysis"].support_decisions,
                        "trail_decisions": snap.payload["analysis"].trail_decisions,
                        "current_level": snap.payload["analysis"].current_level,
                        "assertion_level": snap.payload["analysis"].assertion_level,
                        "killers": {
                            str(v): k
                            for v, k in snap.payload["analysis"].killers.items()
                        },
                    },
                }
                for sc, snap in all_conflicts[: args.top]
            ],
            "guesses": [
                {
                    "puzzle": snap.puzzle_id,
                    "grid": [snap.state.assigned[c] for c in CELLS],
                    "domains": {
                        str(c): sorted(snap.state.domains[c])
                        for c in CELLS
                        if not snap.state.assigned[c]
                    },
                    "score": sc,
                }
                for sc, snap in all_guesses[: args.top]
            ],
        }
        with open(args.dump, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.dump}", file=sys.stderr)


if __name__ == "__main__":
    main()
