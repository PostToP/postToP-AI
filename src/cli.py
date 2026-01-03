import sys

import log


def run_fetch() -> None:
    from data.database import main as fetch_videos

    fetch_videos()


def run_preprocess() -> None:
    from data.preprocess import preprocess_dataset

    preprocess_dataset()


def run_train() -> None:
    from model.train import create_model

    create_model()


def run_optuna() -> None:
    from model.optuna import main as optuna_main

    optuna_main()


def main() -> None:
    COMMANDS = {
        "fetch": run_fetch,
        "preprocess": run_preprocess,
        "train": run_train,
        "optuna": run_optuna,
    }

    if len(sys.argv) < 2:
        print("Usage: python cli.py <operations>")
        print(f"Available operations: {', '.join(COMMANDS.keys())}")
        return

    commands = sys.argv[1:]

    for i in commands:
        if i not in COMMANDS:
            print(f"Unknown command: {i}")
            return

    while commands:
        command = commands.pop(0)
        COMMANDS[command]()


if __name__ == "__main__":
    main()
