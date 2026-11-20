from pathlib import Path
from streamlit.testing.v1 import AppTest
import time

print("Initializing AppTest...")
t0 = time.time()
script_path = str(Path(__file__).parent.parent / "app" / "streamlit_app.py")
at = AppTest.from_file(script_path, default_timeout=10.0)
print(f"Loaded in {time.time() - t0:.2f}s, running...")
t1 = time.time()
at.run()
print(f"Ran in {time.time() - t1:.2f}s")
print("Exceptions:", [e.value for e in at.exception])
print("Tabs found:", len(at.tabs))
