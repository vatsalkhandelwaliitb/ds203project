"""Restart a real Python kernel and execute the notebook; optionally recreate generated outputs.

First run `python verify_delivery.py` to verify the delivered bytes.
Then: python execute_notebook.py --clean-outputs
"""
import argparse
import asyncio
import atexit
import base64
import os
import shutil
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager

ROOT = Path(__file__).resolve().parent


def main():
    # ZeroMQ needs selector support on Windows; the default Proactor loop
    # requires a helper thread that can fail during interpreter shutdown.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-outputs", action="store_true")
    parser.add_argument("--in-process", action="store_true", help="Socket-free fallback: execute with IPython in this fresh Python process, not a Jupyter kernel.")
    args = parser.parse_args()
    if args.clean_outputs:
        # Preserve verification evidence and the tested environment record.
        for name in ["outputs/figures", "outputs/tables", "data/processed"]:
            path = ROOT / name
            if path.is_symlink():
                raise ValueError(f"Refusing to clean a symlink: {path}")
            if path.exists(): shutil.rmtree(path)
            path.mkdir(parents=True)
        (ROOT / "outputs/results.json").unlink(missing_ok=True)
    path = ROOT / "Nation_Clustering.ipynb"
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    if args.in_process:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        os.chdir(ROOT)
        shell = InteractiveShell.instance()
        count = 0
        for cell in nb.cells:
            if cell.cell_type != "code": continue
            count += 1
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(cell.source, store_history=False)
            if not result.success:
                raise RuntimeError(f"Notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}")
            cell.execution_count = count
            for name, value in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
                if value: cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
            for item in captured.outputs:
                data = {mime: base64.b64encode(value).decode() if isinstance(value, bytes) else value for mime, value in item.data.items()}
                cell.outputs.append(nbformat.v4.new_output("display_data", data=data, metadata=item.metadata or {}))
        mode = "fresh Python process using IPython (socket-free fallback, not a Jupyter kernel)"
    else:
        km = AsyncKernelManager(kernel_name="python3", ip="127.0.0.1")
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(nb, timeout=180, kernel_name="python3", km=km,
                                resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute(cleanup_kc=True)
        except Exception:
            # nbclient can register an atexit hook before startup fails; avoid a second cleanup error.
            atexit.unregister(client._cleanup_kernel)
            if km.has_kernel: client._cleanup_kernel()
            raise
        mode = "fresh Jupyter kernel"
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code)
    assert all(o.output_type != "error" for c in code for o in c.outputs)
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print(f"PASS: executed {len(code)} code cells in a {mode} with Python {sys.version.split()[0]}.")
    print("Generated outputs were recreated from empty folders." if args.clean_outputs else "Outputs regenerated.")


if __name__ == "__main__":
    main()
