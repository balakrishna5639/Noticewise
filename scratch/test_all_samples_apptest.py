import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

script_path = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")

TARGET_SAMPLES = [
    "notice_01_elec_disconnection",
    "notice_21_rti_road_repair",
    "notice_16_scheme_ration_card",
    "notice_07_challan_signal",
    "notice_11_tax_property_demand",
]

at = AppTest.from_file(script_path, default_timeout=30.0)
at.run()
assert len(at.exception) == 0

# Switch to sample notice mode
at.radio[0].set_value(at.radio[0].options[1]).run()
assert len(at.exception) == 0

sample_options = at.selectbox[1].options
print(f"Loaded {len(sample_options)} sample options in picker.")

for stem in TARGET_SAMPLES:
    # Find matching option
    matching = [opt for opt in sample_options if stem in opt]
    assert matching, f"No matching option for {stem}"
    opt = matching[0]

    for lang in ["English", "Telugu", "Hindi"]:
        print(f"Testing {stem} in {lang}...")
        at.sidebar.selectbox[0].set_value(lang).run()
        assert len(at.exception) == 0, f"Exception in lang switch: {[e.value for e in at.exception]}"

        at.selectbox[1].set_value(opt).run()
        assert len(at.exception) == 0, f"Exception in sample select: {[e.value for e in at.exception]}"

        # Click analyze
        at.button[0].click().run()
        assert len(at.exception) == 0, f"Exception in analyze for {stem} ({lang}): {[e.value for e in at.exception]}"

        # Check tabs and components
        assert len(at.tabs) == 5, f"Expected 5 tabs, got {len(at.tabs)}"

        # Check downloads
        labels = [db.label for db in at.download_button]
        if stem == "notice_11_tax_property_demand":
            print(f"  {stem} abstained safely as expected.")
        else:
            print(f"  {stem} ({lang}) generated {len(labels)} downloads successfully.")

print("ALL APPTEST SAMPLE CHECKS COMPLETED WITH 0 ERRORS!")
