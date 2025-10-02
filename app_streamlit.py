import streamlit as st
import datetime as dt
import math
import pandas as pd

# Try to import pydeck; if missing, show a warning but don't crash
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
    trail_condition_series,
)

st.set_page_config(page_title="MTB Ride Options — Manchester", layout="wide")

# Lookups
key_to_name = {l.key: l.name for l in LOCATIONS}
terrain_map = {l.key: set(t.strip() for t in l.terrain.split(",")) for l in LOCATIONS}
loc_lookup = {l.key: (l.lat, l.lon) for l in LOCATIONS}

st.title("MTB Ride Options — Manchester")
today = dt.datetime.now().date()
default_depart = dt.time(8, 0)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Daily knobs")

    # Home / Start location: always include all locations from config
    all_keys = [l.key for l in LOCATIONS]
    default_home_idx = all_keys.index("urmston") if "urmston" in all_keys else 0
    home_key = st.selectbox(
        "Home / Start location",
        all_keys,
        index=default_home_idx,
        format_func=lambda k: key_to_name.get(k, k),
        help="Choose your base for today. Drive times are measured from here."
    )

    depart = st.time_input("Depart time", value=default_depart)

    # Day-to-day cap; curve ball may extend beyond via the +60 rule
    max_drive = st.slider("Max drive (min)", 30, 120, 90, 5)

    terrain_bias = st.slider("Preference (distance ↔ hills)", -1.0, 1.0, 0.6, 0.1)
    tech_bias = st.slider("Preference (chilled ↔ gnar)", -1.0, 1.0, 0.4, 0.1)
    duration = st.slider("Ride duration (hours)", 1.0, 6.0, 2.5, 0.5)

    season = st.selectbox("Season", ["auto", "winter", "spring", "summer", "autumn"], index=0)

    include_keys = st.multiselect("Include only (optional)", options=[l.key for l in LOCATIONS], help="If set, only these locations are considered.")
    exclude_keys = st.multiselect("Exclude (optional)", options=[l.key for l in LOCATIONS], help="These locations will be ignored.")

    offer_curve_ball = st.checkbox("Offer curve ball", value=True)
    prox_override = st.checkbox(
        "Let terrain & trail override proximity",
        value=True,
        help="If Terrain & Trail are both strong, don't overly penalize distance."
    )

    st.caption("⚡ Weather and drive estimates update at most once per hour (cached in the backend). Change 'Home' to plan trips (e.g., Lakes or 7stanes).")

    # ---- Advanced scoring weights (hidden by default) ----
    with st.expander("Advanced scoring weights", expanded=False):
        use_defaults = st.checkbox("Use default weights from config", value=True)
        if use_defaults:
            weights = DEFAULT_WEIGHTS.copy()
        else:
            st.write("Adjust the influence of each component (they don't need to sum to 1; we'll normalise).")
            w_weather = st.slider("Weight: Weather", 0.0, 1.0, DEFAULT_WEIGHTS["weather"], 0.05)
            w_trail = st.slider("Weight: Trail", 0.0, 1.0, DEFAULT_WEIGHTS["trail"], 0.05)
            w_prox = st.slider("Weight: Proximity", 0.0, 1.0, DEFAULT_WEIGHTS["proximity"], 0.05)
            w_terr = st.slider("Weight: Terrain fit", 0.0, 1.0, DEFAULT_WEIGHTS["terrain_fit"], 0.05)
            w_sec = st.slider("Weight: Secondary", 0.0, 1.0, DEFAULT_WEIGHTS["secondary"], 0.05)
            # normalise to sum=1
            total = max(1e-6, w_weather + w_trail + w_prox + w_terr + w_sec)
            weights = {
                "weather": w_weather/total,
                "trail": w_trail/total,
                "proximity": w_prox/total,
                "terrain_fit": w_terr/total,
                "secondary": w_sec/total,
            }

# Apply settings / overrides
depart_dt = dt.datetime.combine(today, depart)
season_val = season_from_date(today) if season == "auto" else season
set_home_by_key(home_key, LOCATIONS)
set_tech_bias_override(tech_bias)
set_prox_override(prox_override)
set_weight_override(weights if not use_defaults else None)

# Filter set
locs = [l for l in LOCATIONS if (not include_keys or l.key in include_keys) and (l.key not in exclude_keys)]

# ---------------- Scoring helpers ----------------
def score_all(locs, when_dt, weights_use):
    rows = []
    for loc in locs:
        try:
            r = score_location(loc, when_dt, duration, terrain_bias, max_drive, season_val, tech_bias=tech_bias, weights=weights_use)
        except Exception as e:
            r = {
                "key": loc.key, "name": loc.name, "score": 0.0,
                "components": {"weather": 0, "trail": 0, "proximity": 0, "terrain_fit": 0, "secondary": 0},
                "drive_est_min": 9999,
                "recommend_window": f"{when_dt:%H:%M}–{(when_dt + dt.timedelta(hours=duration)):%H:%M}",
                "notes": [f"Error: {e}"]
            }
        rows.append(r)
    return sorted(rows, key=lambda x: x["score"], reverse=True)

rows_today_all = score_all(locs, depart_dt, weights)
rows_tom_all = score_all(locs, depart_dt + dt.timedelta(days=1), weights)

def in_cap(rows): return [r for r in rows if r.get("drive_est_min", 9999) <= max_drive]
rows_today_ok = in_cap(rows_today_all)
rows_tom_ok = in_cap(rows_tom_all)

def baseline_drive(filtered, allrows):
    if filtered: return filtered[0]['drive_est_min']
    if allrows: return allrows[0]['drive_est_min']
    return max_drive

base_today_drive = baseline_drive(rows_today_ok, rows_today_all)
base_tom_drive   = baseline_drive(rows_tom_ok, rows_tom_all)
curve_extra_limit = 60  # minutes

def pick_curve_ball(rows_all, rows_filtered, bias, tech_bias, baseline_drive_min: int, limit_extra_min: int):
    top_keys = {r['key'] for r in rows_filtered[:5]}
    rest = [r for r in rows_all if r['key'] not in top_keys]
    drive_cap = (baseline_drive_min or 0) + (limit_extra_min or 0)
    rest = [r for r in rest if r.get('drive_est_min', 9999) <= drive_cap]
    if not rest: return None
    min_wx = 70.0
    want_elev = {"hills", "steep", "mod_elev"} if bias > 0 else ({"distance", "flat", "gravel"} if bias < 0 else set())
    want_tech = {"technical", "high_tech", "steep"} if tech_bias > 0 else ({"gravel", "flat", "trail_centre", "distance"} if tech_bias < 0 else set())
    best = None; best_val = -1
    for r in rest:
        tags = terrain_map.get(r["key"], set())
        elev_ok = (not want_elev or (tags & want_elev))
        tech_ok = (not want_tech or (tags & want_tech))
        if r["components"]["weather"] >= min_wx and elev_ok and tech_ok:
            val = r["components"]["weather"] * 1.0 + r["score"] * 0.2
            if val > best_val: best, best_val = r, val
    return best or (max(rest, key=lambda x: x["components"]["weather"]) if rest else None)

curve_today = pick_curve_ball(rows_today_all, rows_today_ok, terrain_bias, tech_bias, base_today_drive, curve_extra_limit) if offer_curve_ball else None
curve_tom   = pick_curve_ball(rows_tom_all,   rows_tom_ok,   terrain_bias, tech_bias, base_tom_drive,   curve_extra_limit) if offer_curve_ball else None

# ---------------- Map helpers ----------------
def build_features(rows, exclude_key=None):
    feats = []
    for idx, r in enumerate(rows, start=1):
        if exclude_key and r['key'] == exclude_key: continue
        latlon = loc_lookup.get(r["key"])
        if not latlon:
            continue
        lat, lon = latlon
        radius_m = 800 + 45 * r["score"]
        if idx == 1: color = [0, 170, 0, 215]
        elif idx == 2: color = [255, 200, 0, 215]
        elif idx == 3: color = [30, 144, 255, 215]
        else: color = [128, 0, 128, 215]
        comps = r["components"]
        feats.append({
            "rank": idx, "name": r["name"], "score": r["score"],
            "lat": lat, "lon": lon, "radius": radius_m, "color": color,
            "weather": comps["weather"], "trail": comps["trail"],
            "proximity": comps["proximity"], "terrain_fit": comps["terrain_fit"],
            "secondary": comps["secondary"], "window": r["recommend_window"],
            "drive": r["drive_est_min"],
        })
    return feats

def build_features_out_of_range(rows, exclude_key=None):
    feats = []
    for r in rows:
        if exclude_key and r.get('key') == exclude_key: continue
        lat, lon = loc_lookup.get(r.get("key"), (None, None))
        if lat is None: continue
        radius_m = 800 + 45 * r.get("score", 50)
        feats.append({
            "rank": "X", "name": r.get("name", "out-of-range"), "score": r.get("score", 0),
            "lat": lat, "lon": lon, "radius": radius_m, "color": [128, 0, 128, 0],
            "weather": r.get("components",{}).get("weather", 0), "trail": r.get("components",{}).get("trail", 0),
            "proximity": r.get("components",{}).get("proximity", 0), "terrain_fit": r.get("components",{}).get("terrain_fit", 0),
            "secondary": r.get("components",{}).get("secondary", 0), "window": r.get("recommend_window", ""),
            "drive": r.get("drive_est_min", 0),
        })
    return feats

def meters_to_degrees(lat, dx_m, dy_m):
    lat_rad = math.radians(lat)
    dlat = dy_m / 111320.0
    dlon = dx_m / (111320.0 * math.cos(lat_rad) if math.cos(lat_rad) != 0 else 1e-6)
    return dlat, dlon

def triangle_coords(lat, lon, size_m=5000):
    angles = [90, 210, 330]
    coords = []
    for a in angles:
        rad = math.radians(a); dx = size_m * math.cos(rad); dy = size_m * math.sin(rad)
        dlat, dlon = meters_to_degrees(lat, dx, dy)
        coords.append([lon + dlon, lat + dlat])
    coords.append(coords[0]); return [coords]

# --------------- Unified Tabs: Today / Tomorrow / Trail conditions ---------------
tab_today, tab_tomorrow, tab_trails = st.tabs(["Today", "Tomorrow", "Trail conditions"])

def render_map_and_results(rows_all, rows_ok, base_drive, curve_pick, label):
    # Map first
    if not HAVE_PYDECK:
        st.warning("pydeck is not installed, so the map is hidden. Install with:  \n`python -m pip install pydeck`")
    else:
        exclude_key = curve_pick["key"] if curve_pick else None
        features = build_features(rows_ok, exclude_key=exclude_key)
        out_rows_for_map = [r for r in rows_all if r.get("drive_est_min", 9999) > max_drive]

        if features or curve_pick or out_rows_for_map:
            # Center on top-3 in-range + curve-ball
            top3 = sorted(features, key=lambda x: x.get("rank", 99))[:3] if features else []
            if curve_pick:
                clat, clon = loc_lookup.get(curve_pick["key"], (None, None))
            else:
                clat = clon = None
            cen_lats = [f["lat"] for f in top3]; cen_lons = [f["lon"] for f in top3]
            if clat is not None: cen_lats.append(clat); cen_lons.append(clon)
            avg_lat = sum(cen_lats)/len(cen_lats) if cen_lats else 53.48
            avg_lon = sum(cen_lons)/len(cen_lons) if cen_lons else -2.3

            layers = []
            if features:
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=features,
                    get_position='[lon, lat]',
                    get_radius="radius",
                    get_fill_color="color",
                    pickable=True,
                    stroked=True,
                    get_line_color=[0, 0, 0],
                    line_width_min_pixels=1,
                ))
            if out_rows_for_map:
                out_feats = build_features_out_of_range(out_rows_for_map, exclude_key=exclude_key)
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=out_feats,
                    get_position='[lon, lat]',
                    get_radius="radius",
                    filled=False,
                    stroked=True,
                    get_line_color=[128, 0, 128],
                    line_width_min_pixels=2,
                    pickable=True,
                ))
            if curve_pick is not None:
                clat, clon = loc_lookup.get(curve_pick["key"], (None, None))
                if clat is not None:
                    tri = triangle_coords(clat, clon, size_m=5000)
                    layers.append(pdk.Layer(
                        "PolygonLayer",
                        data=[{"polygon": tri, "name": curve_pick["name"]}],
                        get_polygon="polygon",
                        get_fill_color=[255, 100, 0, 220],
                        get_line_color=[0, 0, 0, 255],
                        line_width_min_pixels=2,
                        pickable=True,
                    ))

            tooltip = {
                "html": ("<b>#{rank} {name}</b><br/>Score: <b>{score}</b><br/>"
                         "Window: {window}<br/>Drive: ~{drive} min<br/><hr style='margin:4px 0'/>"
                         "Weather: {weather}<br/>Trail: {trail}<br/>Terrain fit: {terrain_fit}<br/>Proximity: {proximity}<br/>Secondary: {secondary}"),
                "style": {"backgroundColor": "rgba(30,30,30,0.9)", "color": "white"}
            }
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=7, pitch=0),
                map_style=None,
                tooltip=tooltip,
            )
            st.pydeck_chart(deck, use_container_width=True)

            with st.expander("Legend / colors"):
                st.markdown(
                    "- **#1** green • **#2** yellow • **#3** blue • **#4+** purple\n"
                    "- **Orange triangle** = curve ball (≤ +60 min beyond the #1 filtered drive)\n"
                    "- **Hollow purple circle** = outside current Max drive (not considered in main ranking)"
                )
        else:
            st.info("No locations to display with current filters.")

    # Results under the map
    st.subheader(f"Top picks — {label}")
    for i, r in enumerate(rows_ok[:5], start=1):
        with st.container(border=True):
            st.markdown(f"### {i}) {r['name']} — **{r['score']}**")
            comps = r["components"]
            c = st.columns(5)
            c[0].metric("Weather", comps["weather"])
            c[1].metric("Trail", comps["trail"])
            c[2].metric("Proximity", comps["proximity"])
            c[3].metric("Terrain fit", comps["terrain_fit"])
            c[4].metric("Secondary", comps["secondary"])
            st.write(f"**Window:** {r['recommend_window']} | **Drive:** ~{r['drive_est_min']} min")
            for n in r["notes"]:
                st.write(f"- {n}")

    st.subheader(f"Alternates — {label}")
    for r in rows_ok[5:7]:
        st.write(f"- {r['name']} — {r['score']}")

with tab_today:
    render_map_and_results(rows_today_all, rows_today_ok, base_today_drive, curve_today, "Today")

with tab_tomorrow:
    render_map_and_results(rows_tom_all, rows_tom_ok, base_tom_drive, curve_tom, "Tomorrow")

# ---- Trail conditions tab ----
with tab_trails:
    st.subheader("Trail conditions — last 10 days (higher = drier)")
    days = 10; window = 5
    season_tc = season_val
    data = {}
    for loc in locs:
        series = trail_condition_series(loc, season_tc, days=days, window=window)
        data[key_to_name[loc.key]] = series
    # Build a date index: last 'days' dates up to yesterday
    today_dt = dt.date.today()
    dates = [(today_dt - dt.timedelta(days=(days - i))) for i in range(1, days+1)]
    df = pd.DataFrame(data, index=[d.strftime("%d %b") for d in dates]).T
    st.dataframe(df.style.background_gradient(cmap="Greens", axis=1), use_container_width=True)
    st.caption("Each cell is a trail condition score (0–100) for that day, computed from the preceding 5-day rainfall at that location.")
