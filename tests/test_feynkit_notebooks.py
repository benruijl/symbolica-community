"""Keep every FeynKit tutorial clean and portable."""

import ast
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")


NOTEBOOKS = sorted(
    (Path(__file__).parents[1] / "examples" / "feynkit").glob("[0-9][0-9]_*.ipynb")
)
MARIMO_APPS = sorted(
    (Path(__file__).parents[1] / "examples" / "feynkit").glob("[0-9][0-9]_*_marimo.py")
)


def _missing_annotations(function):
    arguments = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]
    missing = [argument.arg for argument in arguments if argument.annotation is None]
    if function.args.vararg is not None and function.args.vararg.annotation is None:
        missing.append(f"*{function.args.vararg.arg}")
    if function.args.kwarg is not None and function.args.kwarg.annotation is None:
        missing.append(f"**{function.args.kwarg.arg}")
    if function.returns is None:
        missing.append("return")
    return missing


def _is_marimo_cell(function):
    if function.name != "_":
        return False
    return any(
        (isinstance(decorator, ast.Attribute) and decorator.attr == "cell")
        or (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "cell"
        )
        for decorator in function.decorator_list
    )


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_feynkit_tutorial(notebook_path):
    """Tutorials use a generic kernel and do not commit transient output."""

    notebook = nbformat.read(notebook_path, as_version=4)
    assert notebook.metadata.kernelspec.name == "python3"
    optional_extra = notebook.metadata.get("feynkit", {}).get("optional_extra")
    if notebook_path.name == "04_ufo_loading.ipynb":
        assert optional_extra == "feynkit-ufo"
    else:
        assert optional_extra is None
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(not cell.outputs for cell in code_cells)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_jupyter_tutorial_helpers_are_fully_typed(notebook_path):
    """Functions written for readers document every input and return value."""

    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert not _missing_annotations(node), node.name


def test_every_jupyter_tutorial_has_a_native_marimo_port():
    """Keep the two tutorial formats paired by a predictable basename."""

    expected = {f"{notebook.stem}_marimo.py" for notebook in NOTEBOOKS}
    assert {app.name for app in MARIMO_APPS} == expected


@pytest.mark.parametrize("app_path", MARIMO_APPS, ids=lambda path: path.stem)
def test_feynkit_marimo_app_is_native_and_portable(app_path):
    """Marimo ports stay importable Python apps without Jupyter-only hooks."""

    source = app_path.read_text()
    ast.parse(source)
    assert any(
        f'app = marimo.App(width="{width}")' in source
        for width in ("medium", "full")
    )
    assert "table = partial(mo.ui.table, selection=None)" in source
    assert source.count("mo.ui.table") == 1
    assert "pagination=False" not in source
    assert "show_download=False" not in source
    assert "@app.cell" in source
    assert 'if __name__ == "__main__":' in source
    assert "get_ipython" not in source
    assert ".ipynb_checkpoints" not in source


@pytest.mark.parametrize("app_path", MARIMO_APPS, ids=lambda path: path.stem)
def test_marimo_tutorial_helpers_are_fully_typed(app_path):
    """Reader-facing helpers are typed; Marimo's generated cells are exempt."""

    tree = ast.parse(app_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _is_marimo_cell(
            node
        ):
            assert not _missing_annotations(node), node.name
