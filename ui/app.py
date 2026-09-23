from pathlib import Path
import sys

import streamlit as st

# Ensure page config is called as the first Streamlit command
st.set_page_config(
    page_title="TactEar // Tactical Acoustic Intelligence",
    page_icon="⌖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactear.ui.app import main

if __name__ == "__main__":
    main()
