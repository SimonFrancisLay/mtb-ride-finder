MTB Ride Options Agent (Manchester)
===================================

Two ways to run:

1) Command line
----------------
- Ensure Python 3.9+ and `pip install -r requirements.txt`
- Then:
  python mtb_agent.py --depart "08:00" --max-drive 90 --terrain-bias 0.6 --duration 2.5
  # Back-compat: use --terrain hills|distance instead of --terrain-bias

Optional flags:
  --include "hayfield,marple"   # consider only these
  --exclude "windermere"        # skip this location
  --season "winter"             # override season

2) Streamlit app
-----------------
- Install requirements then run:
  streamlit run app_streamlit.py

Notes
-----
- Weather & precip data from Open-Meteo (no key).
- Drive time uses typical ranges + a rush-hour multiplier (no routing API required).
- Trail dryness is inferred from the last 7 days of precipitation and the location's drainage profile, with a seasonal factor.
- You can tune weights and defaults in config.yaml.

Updates
-------
- Continuous preference via --terrain-bias (-1 distance ↔ +1 hills). Streamlit shows a slider.
- Streamlit shows **Today** and **Tomorrow** tabs (same depart time) with separate rankings.
