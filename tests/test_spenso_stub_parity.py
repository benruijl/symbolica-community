"""Keep the generated Spenso stub and public runtime surface in sync."""

import ast
from pathlib import Path

import symbolica.community.spenso as spenso
from symbolica.community import spenso_native


STUB_PATH = (
    Path(__file__).parents[1]
    / "python"
    / "symbolica"
    / "community"
    / "spenso"
    / "__init__.pyi"
)
STRUCTURED_API_EXPORTS = {
    "_",
    "AUTO",
    "BroadcastFunction",
    "TensorExpression",
    "TensorFunctionLibrary",
    "TensorNetwork",
    "as_tensor",
    "chain",
    "dot",
    "format_tensor",
    "formatted",
    "to_typst",
    "trace",
}


def declared_stub_exports() -> set[str]:
    """Collect declarations, excluding imports used only for annotations."""
    exports: set[str] = set()
    for node in ast.parse(STUB_PATH.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            exports.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            exports.add(node.target.id)
        elif isinstance(node, ast.Assign):
            exports.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return {name for name in exports if not name.startswith("_") or name == "_"}


def stub_methods(class_name: str, method_name: str) -> list[ast.FunctionDef]:
    tree = ast.parse(STUB_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]


def annotation(node: ast.expr | None) -> str:
    assert node is not None
    return ast.unparse(node).replace("builtins.", "")


def test_spenso_runtime_and_stub_exports_match():
    runtime_exports = {
        name for name in dir(spenso) if not name.startswith("_") or name == "_"
    }

    assert runtime_exports == declared_stub_exports()
    assert set(spenso_native.__all__) == runtime_exports
    assert "initialize_module" not in spenso_native.__all__
    assert STRUCTURED_API_EXPORTS <= runtime_exports
    assert {
        "TensorStructure",
        "TensorIndices",
        "LibraryTensor",
        "TensorNamespace",
    }.isdisjoint(
        runtime_exports
    )


def test_structured_stub_overloads_capture_runtime_return_types():
    broadcast_calls = stub_methods("BroadcastFunction", "__call__")
    assert {annotation(method.returns) for method in broadcast_calls} == {
        "Expression",
        "TensorExpression",
        "TensorNetwork",
    }
    assert any(
        "TensorExpression" in annotation(method.args.args[-1].annotation)
        for method in broadcast_calls
        if annotation(method.returns) == "TensorExpression"
    )

    expression_getitems = stub_methods("TensorExpression", "__getitem__")
    assert {annotation(method.returns) for method in expression_getitems} == {
        "int",
        "list[int]",
        "list[list[int]]",
    }
