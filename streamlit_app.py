"""
WarnaIn - Root entrypoint for Streamlit Community Cloud.
Automatically routes execution to app/app.py.
"""
import os
import runpy
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

target_app = os.path.join(PROJECT_ROOT, "app", "app.py")
runpy.run_path(target_app, run_name="__main__")
