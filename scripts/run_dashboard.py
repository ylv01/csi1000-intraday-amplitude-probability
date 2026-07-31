"""Launch the local Streamlit dashboard.

The SSL fallback only activates when the Windows certificate store cannot be
loaded.  It keeps Streamlit/Tornado usable by loading certifi's CA bundle.
"""

from __future__ import annotations

import runpy
import ssl
import sys
from pathlib import Path


def _configure_ssl_fallback() -> None:
    if sys.platform != "win32":
        return

    try:
        ssl.create_default_context()
        return
    except ssl.SSLError:
        import certifi

        ca_bundle = certifi.where()

        def load_certifi_certs(
            context: ssl.SSLContext,
            purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
        ) -> None:
            del purpose
            context.load_verify_locations(cafile=ca_bundle)

        ssl.SSLContext.load_default_certs = load_certifi_certs


def main() -> None:
    _configure_ssl_fallback()
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    runpy.run_module("streamlit", run_name="__main__")


if __name__ == "__main__":
    main()
