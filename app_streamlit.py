import streamlit as st
import os, yaml, datetime as dt

st.set_page_config(page_title="MTB Ride Options — Diagnostic", layout="wide")

st.title("MTB Ride Options — Diagnostic Build")

st.subheader("Config sanity")
cwd = os.getcwd()
cfg_present = os.path.exists(os.path.join(cwd, "config.yaml"))
st.write(f"**Working directory:** `{cwd}`")
st.write(f"**config.yaml present:** `{cfg_present}`")
if cfg_present:
    try:
        with open("config.yaml","r") as f:
            cfg_raw = f.read()
        st.code(cfg_raw[:1000] + ("..." if len(cfg_raw)>1000 else ""), language="yaml")
        cfg = yaml.safe_load(cfg_raw) or {}
        locs = cfg.get("locations", [])
        st.success(f"Parsed YAML OK. Locations found: {len(locs)}")
        if not locs:
            st.error("No locations in YAML. Please ensure 'locations:' list is populated.")
    except Exception as e:
        st.exception(e)
        st.stop()
else:
    st.error("config.yaml not found in the repo root (same folder as app_streamlit.py).")
    st.stop()

st.subheader("mtb_agent import")
try:
    from mtb_agent import LOCATIONS, DEFAULT_WEIGHTS, season_from_date
    st.write(f"Imported mtb_agent. LOCATIONS: {len(LOCATIONS)}; default weights: {DEFAULT_WEIGHTS}")
except Exception as e:
    st.exception(e)
    st.stop()

st.success("Diagnostics passed — your main app should render if this page works.")

st.info("Tip: Commit both **config.yaml** and **mtb_agent.py** at repo root. In Streamlit Cloud, go to 'Manage app' → 'Logs' → 'App logs' to see runtime errors.")
