"""Historical experiment: requires its original checkout/environment.

Execute the NNsight experiment notebooks after successful regeneration."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


root = Path(__file__).resolve().parents[2]
for name in ("role_interventions", "baseline_accuracy"):
    path = root / "notebook" / f"{name}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    nbformat.write(notebook, path)
    print(f"Executed {path.relative_to(root)}", flush=True)
