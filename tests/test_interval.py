"""Check the meeting-scheduling example of the advanced-modelling chapter."""

from itertools import pairwise

from _snippets import run_snippet


def test_meetings_do_not_overlap():
    """Every meeting is scheduled within its windows and no two overlap."""
    namespace = run_snippet("advanced_modelling/meeting_solution")
    meetings, scheduled = namespace["meetings"], namespace["scheduled_times"]
    assert scheduled.keys() == meetings.keys()
    slots = sorted((scheduled[name], meetings[name].duration) for name in meetings)
    for (start, duration), (next_start, _) in pairwise(slots):
        assert start + duration <= next_start
    for name, info in meetings.items():
        assert any(lb <= scheduled[name] <= ub for lb, ub in info.start_times)
