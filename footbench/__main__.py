"""Footbench CLI: one subcommand per pipeline stage, plus `run` and `probe`."""

from __future__ import annotations

import argparse

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="footbench", description="Footbench evaluation pipeline")
    parser.add_argument(
        "command",
        choices=["probe", "generate", "sandbox", "check", "judge", "aggregate", "publish", "run"],
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="MODEL",
        help="restrict probe/generate (candidates) or judge (judges) to these model names",
    )
    parser.add_argument("--instance", metavar="ID", help="restrict to one instance id")
    parser.add_argument(
        "--dump-prompt",
        action="store_true",
        help="judge: print the assembled prompt instead of calling the API",
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
        elif command == "judge":
            from . import judge

            judge.run(
                cfg,
                only_judges=args.only,
                only_instance=args.instance,
                dump_prompt=args.dump_prompt,
            )
        elif command == "aggregate":
            from . import aggregate

            aggregate.run(cfg)
        elif command == "publish":
            from . import publish

            publish.run(cfg)

    if args.command == "run":
        for step in ("generate", "sandbox", "check", "judge", "aggregate", "publish"):
            print(f"=== {step} ===")
            do(step)
    else:
        do(args.command)


if __name__ == "__main__":
    main()
