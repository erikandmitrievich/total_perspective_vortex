BASE_GROUPS = (([3, 7, 11],  "left fist vs right fist (executed)"),
               ([4, 8, 12],  "left fist vs right fist (imagined)"),
               ([5, 9, 13],  "both fists vs both feet (executed)"),
               ([6, 10, 14], "both fists vs both feet (imagined)"))
"""
The four task groups, as ``(runs, label)`` pairs.

Each group is three repetitions of one task. Within a group T1 and T2 have
one meaning throughout, which is what makes concatenation across the three
runs legitimate.
"""

EXPERIMENTS = BASE_GROUPS + (
    ([3, 4, 7, 8, 11, 12],  "left fist vs right fist (pooled)"),
    ([5, 6, 9, 10, 13, 14], "both fists vs both feet (pooled)"))
"""
The six experiments scored by the sweep.

The last two pool an executed group with its imagined counterpart: the
movements behind T1/T2 are identical, only the execution modality differs,
so the labels remain consistent and the epoch count doubles. Pooled
experiments exist in this namespace only — they are not reachable from the
CLI, see ``runs_for``.
"""

SUBJECTS = range(1, 110)
"""Subject numbers the sweep iterates, in ascending order."""


def runs_for(run: int) -> list[int]:
    """
    Expand a run number to its base group.

    A single run is not a self-contained binary problem: the three runs of
    a group are repetitions of one task and are the unit that gets trained
    on. Only ``BASE_GROUPS`` is searched — a run number cannot disambiguate
    between its base group and the pooled experiment containing it, so the
    pooled experiments are deliberately unreachable from here.

    Parameters
    ----------
    run : int
        Motor-imagery run number: 3-14, excluding the baselines 1-2.

    Returns
    -------
    runs : list of int
        The three runs of the group containing ``run``.

    Raises
    ------
    ValueError
        If ``run`` is not one of the twelve motor-imagery runs.
    """
    for runs, _ in BASE_GROUPS:
        if run in runs:
            return runs
    valid = sorted(r for runs, _ in BASE_GROUPS for r in runs)
    raise ValueError(f"run {run} is not a motor-imagery run (valid: {valid})")
