"""Footbench CLI: one subcommand per pipeline stage, plus `run` and `probe`."""

from __future__ import annotations

import argparse

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="footbench", description="Footbench evaluation pipeline")
    parser.add_argument(
        "command",
        choices=[
            "probe",
            "generate",
            "sandbox",
            "check",
            "tables",
            "pairwise",
            "bradley-terry",
            "run",
        ],
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        help="pairwise only: estimate | submit | deepseek | status | collect | csv",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="MODEL",
        help="restrict probe/generate (candidates) or pairwise (judges) to these model names",
    )
    parser.add_argument("--instance", metavar="ID", help="restrict to one instance id")
    parser.add_argument(
        "--dump-prompt",
        action="store_true",
        help="pairwise: print the assembled prompt instead of calling the API",
    )
    args = parser.parse_args(argv)
    cfg = load_config()

    def do(command: str) -> None:
        if command == "probe":
            from . import providers

            providers.probe_cli(cfg, only=args.only)
        elif command == "generate":
            from . import generate

            generate.run(cfg, only=args.only)
        elif command == "sandbox":
            from . import sandbox

            sandbox.run(cfg, only_instance=args.instance)
        elif command == "check":
            from . import checks

            checks.run(cfg, only_instance=args.instance)
        elif command == "tables":
            from . import tables

            tables.run(cfg)
        elif command == "pairwise":
            from . import pairwise

            pairwise.run_cli(cfg, args.subcommand, only=args.only, dump_prompt=args.dump_prompt)
        elif command == "bradley-terry":
            from . import bradley_terry

            bradley_terry.run(cfg)

    if args.command == "run":
        for step in ("generate", "sandbox", "check", "tables"):
            print(f"=== {step} ===")
            do(step)
    else:
        do(args.command)


if __name__ == "__main__":
    main()
