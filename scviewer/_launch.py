"""Console-script launcher for the scviewer Streamlit app.

Exposed as the ``scviewer`` entry point. Forwards any extra CLI args to
Streamlit after ``--`` so ``scviewer --data-dir mydata`` works.
"""
from __future__ import annotations

import os
import sys


def _app_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")


def main() -> None:
    """Launch `streamlit run app.py -- <user args>`."""
    app = _app_path()
    argv = sys.argv[1:]
    # Everything the user passes goes to the app after Streamlit's own `--`.
    stream_args = ["streamlit", "run", app]
    if argv:
        stream_args += ["--"] + argv
    try:
        from streamlit.web import cli as stcli
    except Exception as exc:  # pragma: no cover
        sys.exit(
            "scviewer: Streamlit is required to launch the viewer "
            f"(import failed: {exc}). Install with `pip install streamlit`."
        )
    sys.argv = stream_args
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
