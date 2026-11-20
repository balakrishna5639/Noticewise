from pathlib import Path
from streamlit.testing.v1 import AppTest

script_path = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
at = AppTest.from_file(script_path, default_timeout=25.0)
at.run()
assert len(at.exception) == 0, f"App execution failed: {[e.value for e in at.exception]}"
print("Initial load OK. Tabs:", [t.label for t in at.tabs])

# Switch radio to sample notice picker
print("Switching to sample notice picker...")
at.radio[0].set_value(at.radio[0].options[1]).run()
assert len(at.exception) == 0

print("Buttons available after sample select:", len(at.button))
assert len(at.button) > 0, "Analyze button should appear when sample is selected"

# Click analyze button
print("Clicking analyze...")
at.button[0].click().run()
print("Analyze run finished! Exceptions:", len(at.exception))
for ex in at.exception:
    print("EX:", ex.value)
assert len(at.exception) == 0

print("Download buttons count:", len(at.download_button))
for db in at.download_button:
    print("  Button label:", db.label)
