"""Execute all FeynKit tutorials against the installed package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "feynkit"
OPTIONAL_EXTRA_KEY = "optional_extra"


def optional_extra(notebook_path: Path) -> str | None:
    """Return the explicitly declared optional extra for one tutorial."""

    import nbformat

    notebook = nbformat.read(notebook_path, as_version=4)
    feynkit_metadata = notebook.metadata.get("feynkit", {})
    extra = feynkit_metadata.get(OPTIONAL_EXTRA_KEY)
    if extra is not None and not isinstance(extra, str):
        raise TypeError(
            f"{notebook_path} metadata feynkit.{OPTIONAL_EXTRA_KEY} must be a string"
        )
    return extra


def execute(notebook_path: Path) -> None:
    """Execute one notebook in memory and fail on its first error."""

    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(notebook_path, as_version=4)
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        allow_errors=False,
        record_timing=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    ).execute()


def execute_isolated(notebook_path: Path) -> None:
    """Use a fresh runner so community-license state cannot leak between tutorials."""

    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--one", str(notebook_path)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="also execute tutorials that declare an optional package extra",
    )
    args = parser.parse_args()

    if args.one is not None:
        execute(args.one.resolve())
        return 0

    notebooks = sorted(EXAMPLES.glob("[0-9][0-9]_*.ipynb"))
    if not notebooks:
        raise RuntimeError(f"no FeynKit tutorials found in {EXAMPLES}")
    for notebook_path in notebooks:
        extra = optional_extra(notebook_path)
        if extra is not None and not args.include_optional:
            print(
                f"skipped {notebook_path.relative_to(ROOT)} "
                f"(install symbolica[{extra}])"
            )
            continue
        execute_isolated(notebook_path)
        print(f"executed {notebook_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
