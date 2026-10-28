"""
Convenience entry point for Streamlit execution:
Run via: streamlit run WarnaIn3/streamlit_app.py
"""

import os
import sys
import runpy

V3_ROOT = os.path.abspath(os.path.dirname(__file__))
if V3_ROOT not in sys.path:
    sys.path.insert(0, V3_ROOT)

runpy.run_path(os.path.join(V3_ROOT, "app", "app.py"), run_name="__main__")
