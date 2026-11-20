import os
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

script_path = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
at = AppTest.from_file(script_path, default_timeout=20.0)
at.run()
print("Run finished!")
print("Exception count:", len(at.exception))
for ex in at.exception:
    print("EX:", ex.value)
print("Tabs rendered:", [t.label for t in at.tabs])
