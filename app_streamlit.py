import streamlit as st
import datetime as dt
import math
import pandas as pd

try:
    import pydeck as pdk
    HAVE_PYDECK = True
except Exception:
    HAVE_PYDECK = False

from mtb_agent import (
    LOCATIONS,
    DEFAULT_WEIGHTS,
    season_from_date,
    score_location,
    set_tech_bias_override,
    set_prox_override,
    set_home_by_key,
    set_weight_override,
    trail_condition_series_legacy,   # <<< use legacy here
    trail_condition_for_date_outlook,
)

st.set_page_config(page_title="MTB Ride Options — Legacy History Restored", layout="wide")

if not LOCATIONS:
    st.error("No locations found. Make sure **config.yaml** is present in the repo root and valid.")
    st.stop()

key_to_name = {l.key: l.name for l in LOCATIONS}
key_to_region = {l.key: getattr(l, "region", "") for l in LOCATIONS}
terrain_map = {l.key: set(t.strip() for t in l.terrain.split(",")) for l in LOCATIONS}
loc_lookup = {l.key: (l.lat, l.lon) for l in LOCATIONS}

st.title("MTB Ride Options — Legacy History Restored")
today = dt.date.today()
default_depart = dt.time(8, 0)

def fmt_date(d: dt.date):
    if d == today: return "Today"
    if d == today + dt.timedelta(days=1): return "Tomorrow"
    return d.strftime("%a %d %b")

with st.sidebar:
    st.header("Daily knobs")
    regions_sorted = sorted({r for r in key_to_region.values() if r})
    region_sel = st.multiselect("Region(s)", options=regions_sorted, help="Pick one or more regions to include. Leave empty for all.")
    all_keys = [l.key for l in LOCATIONS]
    default_home_idx = all_keys.index("urmston") if "urmston" in all_keys else 0
    home_key = st.selectbox("Home / Start location", all_keys, index=default_home_idx, format_func=lambda k: key_to_name.get(k, k))
    depart = st.time_input("Depart time", value=default_depart)
    options = [today + dt.timedelta(days=i) for i in range(0, 15)]
    ride_date = st.selectbox("Ride date", options=options, index=0, format_func=fmt_date)
    max_drive = st.slider("Max drive (min)", 30, 120, 90, 5)
    terrain_bias = st.slider("Preference (distance ↔ hills)", -1.0, 1.0, 0.6, 0.1)
    tech_bias = st.slider("Preference (chilled ↔ gnar)", -1.0, 1.0, 0.4, 0.1)
    duration = st.slider("Ride duration (hours)", 1.0, 6.0, 2.5, 0.5)
    season = st.selectbox("Season", ["auto", "winter", "spring", "summer", "autumn"], index=0)
    include_keys = st.multiselect("Include only (optional)", options=[l.key for l in LOCATIONS])
    exclude_keys = st.multiselect("Exclude (optional)", options=[l.key for l in LOCATIONS])
    offer_curve_ball = st.checkbox("Offer curve ball", value=True)
    prox_override = st.checkbox("Let terrain & trail override proximity", value=True)
    with st.expander("Advanced scoring & options", expanded=False):
        use_defaults = st.checkbox("Use default weights from config", value=True)
        if use_defaults:
            weights = DEFAULT_WEIGHTS.copy()
        else:
            w_weather = st.slider("Weight: Weather", 0.0, 1.0, DEFAULT_WEIGHTS["weather"], 0.05)
            w_trail = st.slider("Weight: Trail", 0.0, 1.0, DEFAULT_WEIGHTS["trail"], 0.05)
            w_prox = st.slider("Weight: Proximity", 0.0, 1.0, DEFAULT_WEIGHTS["proximity"], 0.05)
            w_terr = st.slider("Weight: Terrain fit", 0.0, 1.0, DEFAULT_WEIGHTS["terrain_fit"], 0.05)
            w_sec = st.slider("Weight: Secondary", 0.0, 1.0, DEFAULT_WEIGHTS["secondary"], 0.05)
            total = max(1e-6, w_weather + w_trail + w_prox + w_terr + w_sec)
            weights = {k: v/total for k,v in {"weather":w_weather,"trail":w_trail,"proximity":w_prox,"terrain_fit":w_terr,"secondary":w_sec}.items()}

depart_dt = dt.datetime.combine(ride_date, depart)
days_ahead = (ride_date - today).days
trail_mode_eff = "time_aware" if days_ahead <= 1 else "daily"
season_val = season_from_date(ride_date) if season == "auto" else season
set_home_by_key(home_key, LOCATIONS)
set_tech_bias_override(tech_bias)
set_prox_override(prox_override)
set_weight_override(weights if not use_defaults else None)

def in_regions(k: str) -> bool:
    if not region_sel: return True
    return key_to_region.get(k, "") in region_sel

locs = [l for l in LOCATIONS if in_regions(l.key) and (not include_keys or l.key in include_keys) and (l.key not in exclude_keys)]

def score_all(locs, when_dt, weights_use, trail_mode_key):
    rows = []
    for loc in locs:
        try:
            r = score_location(loc, when_dt, duration, terrain_bias, max_drive, season_from_date(when_dt.date()), 
                               tech_bias=tech_bias, weights=weights_use, trail_mode=trail_mode_key)
        except Exception as e:
            r = {"key": loc.key, "name": loc.name, "score": 0.0, "components": {"weather":0,"trail":0,"proximity":0,"terrain_fit":0,"secondary":0},
                 "drive_est_min": 9999, "recommend_window": f"{when_dt:%H:%M}–{(when_dt + dt.timedelta(hours=duration)):%H:%M}", "notes": [f"Error: {e}"]}
        rows.append(r)
    return sorted(rows, key=lambda x: x["score"], reverse=True)

rows_all = score_all(locs, depart_dt, weights, trail_mode_eff)
rows_ok = [r for r in rows_all if r.get("drive_est_min", 9999) <= max_drive]

# ---- UI (map + results) omitted for brevity: keep your current one ----
st.subheader(f"Top picks for {( 'Today' if ride_date==dt.date.today() else ride_date.strftime('%a %d %b') )}")
st.write("Top picks and Outlook still use the time-aware planner. (Map/results unchanged from your working app.)")

# ---- Trail conditions: History uses LEGACY again ----
st.markdown("---")
st.header("Trail conditions")

with st.expander("History (last 10 days) — daily aggregates (legacy model restored)", expanded=True):
    days = 10
    season_tc = season_val
    data_hist = {}
    for loc in locs:
        series = trail_condition_series_legacy(loc, season_tc, days=days, window=5)
        data_hist[key_to_name[loc.key]] = series
    today_dt = dt.date.today()
    dates = [(today_dt - dt.timedelta(days=(days - i))) for i in range(1, days+1)]
    df = pd.DataFrame(data_hist, index=[d.strftime("%d %b") for d in dates]).T
    try:
        styled = df.style.background_gradient(cmap="Greens", axis=1)
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)
    st.caption("Higher = drier. Heavy rain drops scores immediately; recovery depends on drainage, mud sensitivity, and season caps.")
