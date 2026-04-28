from __future__ import annotations

from .cli.run_highdimension_cli import main


def _cli() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(_cli())
