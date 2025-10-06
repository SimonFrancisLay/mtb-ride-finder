# app_streamlit.py placeholder — please paste your current working copy here.
import streamlit as st
st.write('App placeholder: please use the previous working app_streamlit.py')


# ---- SAFETY GUARD: stop early if config has no locations ----
try:
    from mtb_agent import LOCATIONS as _LOC_TEST
    if not _LOC_TEST:
        st.error("No locations found. Make sure **config.yaml** is present in the repo root and valid.")
        st.stop()
except Exception as _e:
    st.error(f"Error loading config/locations: {_e}. Ensure **config.yaml** exists and is valid.")
    st.stop()
