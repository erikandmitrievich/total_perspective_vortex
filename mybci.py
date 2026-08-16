import sys


def val_argv(argv):
    if len(argv) == 1:
        return {"run_all": True}

    assert len(argv) == 4, "Usage: python mybci.py [subject_id] [task_id] [train/predict]"

    try:
        subj = int(argv[1])
        task = int(argv[2])
        mode = argv[3].lower()

        assert mode in ['train', 'predict'], "Mode must be either 'train' or 'predict'"

        return {
            "run_all": False,
            "subject": subj,
            "task": task,
            "mode": mode
        }
    except ValueError:
        raise AssertionError("Subject and task must be integers.")


def main():
    try:
        args = val_argv(sys.argv)
    except AssertionError as err:
        print(f"AssertionError: {err}")
        return

    if args.get("run_all"):
        print("Running full evaluation across all subjects and experiments...")
        # TODO: call general evaluation logic
    else:
        print(f"Executing: Subject {args['subject']} | Task {args['task']} | Mode {args['mode']}")
        if args['mode'] == 'train':
            pass # TODO: call train.py logic
        elif args['mode'] == 'predict':
            pass # TODO: call predict.py logic


if __name__ == "__main__":
    main()
