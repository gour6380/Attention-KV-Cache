"""Static notebook contract; these tests do not execute notebook cells."""

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def notebook():
    return nbformat.read(ROOT / "notebooks" / "attention_kv_cache.ipynb", as_version=4)


def test_notebook_is_valid_and_uses_project_kernel():
    nb = notebook()
    nbformat.validate(nb)
    assert nb.metadata.kernelspec.name == "attention-kv-cache"
    for cell in nb.cells:
        if cell.cell_type == "code":
            # Executed outputs are published evidence; a fresh notebook is valid too.
            assert not any(output.output_type == "error" for output in cell.outputs)
            ast.parse(cell.source)


def test_each_variant_has_separate_training_evaluation_and_benchmark_cells():
    cells = [cell.source for cell in notebook().cells if cell.cell_type == "code"]
    for function in ("train_variant", "evaluate_variant", "benchmark_variant"):
        invocation_cells = []
        for source in cells:
            calls = [
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function
            ]
            if calls:
                assert len(calls) == 1
                invocation_cells.append(source)
        assert len(invocation_cells) == 3


def test_reporting_and_evaluation_never_call_training():
    for name in ("reporting.py", "evaluate.py", "benchmark.py", "generate.py"):
        tree = ast.parse((ROOT / "src" / name).read_text())
        names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "train_variant" not in names
