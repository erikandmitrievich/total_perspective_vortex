CLASS_NAMES = {
    3:  ("left fist", "right fist"), 4:  ("left fist", "right fist"),
    5:  ("both fists", "both feet"), 6:  ("both fists", "both feet"),
    7:  ("left fist", "right fist"), 8:  ("left fist", "right fist"),
    9:  ("both fists", "both feet"), 10: ("both fists", "both feet"),
    11: ("left fist", "right fist"), 12: ("left fist", "right fist"),
    13: ("both fists", "both feet"), 14: ("both fists", "both feet"),
}

GROUPS = (([3, 7, 11],  "left fist vs right fist (executed)"),
          ([4, 8, 12],  "left fist vs right fist (imagined)"),
          ([5, 9, 13],  "both fists vs both feet (executed)"),
          ([6, 10, 14], "both fists vs both feet (imagined)"))

EXCLUDED = {88, 89, 92, 100}          # non-160 Hz / bad annotations

SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDED]


def runs_for(run: int) -> list[int]:
    """
    Expand a run number to the full group sharing its T1/T2 semantics.

    T1/T2 mean different movements per group, so a group is the largest
    set that can be pooled into one binary problem.
    """
    for runs, _ in GROUPS:
        if run in runs:
            return runs
    valid = sorted(r for runs, _ in GROUPS for r in runs)
    raise ValueError(f"run {run} is not a motor-imagery run (valid: {valid})")
