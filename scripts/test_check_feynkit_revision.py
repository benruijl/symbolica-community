"""Unit tests for the FeynKit manifest and lockfile revision checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_feynkit_revision import locked_revision, pinned_dependency

REPOSITORY = "https://github.com/alphal00p/gammaloop"
REVISION = "a" * 40
OTHER_REVISION = "b" * 40


class FeynkitRevisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def write(self, name: str, contents: str) -> Path:
        path = self.root / name
        path.write_text(contents)
        return path

    def manifest(self, *, revision: str = REVISION) -> Path:
        return self.write(
            "Cargo.toml",
            "[dependencies]\n"
            f'feynkit-py = {{ git = "{REPOSITORY}", rev = "{revision}" }}\n',
        )

    def lockfile(
        self,
        *,
        repository: str = REPOSITORY,
        requested_revision: str = REVISION,
        precise_revision: str = REVISION,
    ) -> Path:
        source = f"git+{repository}?rev={requested_revision}#{precise_revision}"
        return self.write(
            "Cargo.lock",
            "version = 4\n\n"
            "[[package]]\n"
            'name = "feynkit-py"\n'
            'version = "0.1.0"\n'
            f'source = "{source}"\n',
        )

    def test_matching_manifest_and_lockfile(self) -> None:
        repository, revision = pinned_dependency(self.manifest())
        self.assertEqual(repository, REPOSITORY)
        self.assertEqual(
            locked_revision(self.lockfile(), repository, revision), REVISION
        )

    def test_manifest_requires_a_full_hexadecimal_pin(self) -> None:
        with self.assertRaisesRegex(ValueError, "full lowercase hexadecimal"):
            pinned_dependency(self.manifest(revision="not-a-commit"))

    def test_lockfile_rejects_a_stale_precise_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "precise revision"):
            locked_revision(
                self.lockfile(precise_revision=OTHER_REVISION),
                REPOSITORY,
                REVISION,
            )

    def test_lockfile_rejects_a_stale_requested_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested revision"):
            locked_revision(
                self.lockfile(requested_revision=OTHER_REVISION),
                REPOSITORY,
                REVISION,
            )

    def test_lockfile_rejects_a_different_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "repository does not match"):
            locked_revision(
                self.lockfile(repository="https://example.com/gammaloop"),
                REPOSITORY,
                REVISION,
            )

    def test_lockfile_requires_one_feynkit_package(self) -> None:
        lockfile = self.write("Cargo.lock", "version = 4\n")
        with self.assertRaisesRegex(ValueError, "exactly one feynkit-py"):
            locked_revision(lockfile, REPOSITORY, REVISION)


if __name__ == "__main__":
    unittest.main()
