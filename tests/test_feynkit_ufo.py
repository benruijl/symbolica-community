"""Static guarantees for the optional raw-UFO smoke path."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "examples" / "feynkit" / "data" / "ufo_scalars"
REQUIRED_UFO_FILES = {
    "__init__.py",
    "coupling_orders.py",
    "couplings.py",
    "function_library.py",
    "lorentz.py",
    "object_library.py",
    "parameters.py",
    "particles.py",
    "restrict_default.dat",
    "vertices.py",
}


def test_minimal_raw_ufo_fixture_is_complete():
    """Keep every import and the default restriction needed by the loader."""

    assert {
        path.name for path in FIXTURE.iterdir() if path.is_file()
    } == REQUIRED_UFO_FILES


def test_ufo_smoke_path_uses_the_public_optional_api():
    """Keep the check installable via the documented extra and off native APIs."""

    script = (ROOT / "scripts" / "check_feynkit_ufo.py").read_text()
    runner = (ROOT / "scripts" / "run_feynkit_notebooks.py").read_text()
    project = (ROOT / "pyproject.toml").read_text()
    tutorial = (ROOT / "examples" / "feynkit" / "README.md").read_text()

    assert "from symbolica.community.feynkit import UfoLoader" in script
    assert 'UfoLoader(restriction_name="default").load(source)' in script
    assert "feynkit_native" not in script
    assert '"04_ufo_loading.ipynb"' in script
    assert '"--one"' in script
    assert '"--include-optional"' in runner
    assert 'feynkit-ufo = ["ufo-model-loader>=0.1.6"]' in project
    assert "pip install 'symbolica[feynkit-ufo]'" in tutorial
