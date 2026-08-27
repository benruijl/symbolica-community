"""Fail when symbolica-community's FeynKit pin trails its integration branch."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

import tomllib

DEFAULT_REPOSITORY = "https://github.com/alphal00p/gammaloop"
FULL_REVISION = re.compile(r"[0-9a-f]{40}\Z")


def _normalized_repository(repository: str) -> str:
    """Normalize equivalent HTTPS Git repository spellings."""

    repository = repository.rstrip("/")
    if repository.endswith(".git"):
        repository = repository[:-4]
    return repository


def pinned_dependency(manifest: Path) -> tuple[str, str]:
    """Read the FeynKit Git repository and exact revision from a manifest."""

    dependency = tomllib.loads(manifest.read_text())["dependencies"]["feynkit-py"]
    if not isinstance(dependency, dict):
        raise TypeError("feynkit-py must be a detailed Git dependency")

    repository = dependency.get("git")
    revision = dependency.get("rev")
    if not isinstance(repository, str) or not repository:
        raise ValueError("feynkit-py must specify its Git repository")
    if not isinstance(revision, str) or not FULL_REVISION.fullmatch(revision):
        raise ValueError(
            "feynkit-py must use a full lowercase hexadecimal revision pin"
        )
    return repository, revision


def pinned_revision(manifest: Path) -> str:
    """Read the exact FeynKit git revision from a Cargo manifest."""

    return pinned_dependency(manifest)[1]


def locked_revision(lockfile: Path, repository: str, revision: str) -> str:
    """Verify and return the precise FeynKit revision recorded by Cargo."""

    packages = tomllib.loads(lockfile.read_text()).get("package", [])
    matches = [package for package in packages if package.get("name") == "feynkit-py"]
    if len(matches) != 1:
        message = (
            "Cargo.lock must contain exactly one feynkit-py package; "
            f"found {len(matches)}"
        )
        raise ValueError(message)

    source = matches[0].get("source")
    if not isinstance(source, str) or not source.startswith("git+"):
        raise ValueError("Cargo.lock feynkit-py must resolve from a Git source")

    parsed = urlsplit(source.removeprefix("git+"))
    locked_repository = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    requested_revisions = parse_qs(parsed.query).get("rev", [])

    if _normalized_repository(locked_repository) != _normalized_repository(repository):
        raise ValueError(
            "Cargo.lock feynkit-py repository does not match Cargo.toml: "
            f"{locked_repository!r} != {repository!r}"
        )
    if requested_revisions != [revision]:
        raise ValueError(
            "Cargo.lock feynkit-py requested revision does not match Cargo.toml: "
            f"{requested_revisions!r} != {[revision]!r}"
        )
    if parsed.fragment != revision:
        raise ValueError(
            "Cargo.lock feynkit-py precise revision does not match Cargo.toml: "
            f"{parsed.fragment!r} != {revision!r}"
        )
    return parsed.fragment


def branch_revision(repository: str, branch: str) -> str:
    """Resolve one remote branch without cloning its repository."""

    output = subprocess.run(
        ["git", "ls-remote", repository, f"refs/heads/{branch}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not output:
        raise RuntimeError(f"remote branch {branch!r} was not found")
    revision, reference = output.split()
    if reference != f"refs/heads/{branch}" or not FULL_REVISION.fullmatch(revision):
        raise RuntimeError(f"unexpected ls-remote output: {output!r}")
    return revision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    parser.add_argument("--lockfile", type=Path, default=Path("Cargo.lock"))
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--branch", default="feynkit")
    args = parser.parse_args()

    dependency_repository, pinned = pinned_dependency(args.manifest)
    locked_revision(args.lockfile, dependency_repository, pinned)
    latest = branch_revision(args.repository, args.branch)
    if pinned == latest:
        print(f"FeynKit is current at {pinned}")
        return 0

    print(
        "::error title=FeynKit update available::"
        f"Pinned {pinned}; refs/heads/{args.branch} is {latest}. "
        "Review the FeynKit diff and run the Rust, wheel, API, and notebook checks "
        "before advancing the pin."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
