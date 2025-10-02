import streamlit as st
import datetime as dt
import math
import pydeck as pdk
from mtb_agent import (
    LOCATIONS,
    season_from_date,
    score_location,
    set_tech_bias_override,
    set_prox_override,
)

st.set_page_config(page_title="MTB Ride Options — Manchester", layout="wide")

# Terrain tags lookup and lat/lon for mapping
terrain_map = {l.key: set(t.strip() for t in l.terrain.split(",")) for l in LOCATIONS}
loc_lookup = {l.key: (l.lat, l.lon) for l in LOCATIONS}

st.title("MTB Ride Options — Manchester")
today = dt.datetime.now().date()
default_depart = dt.time(8, 0)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Daily knobs")
    depart = st.time_input("Depart time", value=default_depart)
    # Keep main max-drive slider modest; curve ball can exceed it by rule below
    max_drive = st.slider("Max drive (min)", 30, 360, 90, 5)
    terrain_bias = st.slider("Preference (distance ↔ hills)", -1.0, 1.0, 0.6, 0.1)
    tech_bias = st.slider("Preference (chilled ↔ gnar)", -1.0, 1.0, 0.4, 0.1)
    duration = st.slider("Ride duration (hours)", 1.0, 6.0, 2.5, 0.5)
    season = st.selectbox("Season", ["auto","winter","spring","summer","autumn"], index=0)
    include_keys = st.multiselect("Include only (optional)", options=[l.key for l in LOCATIONS])
    exclude_keys = st.multiselect("Exclude (optional)", options=[l.key for l in LOCATIONS])
    include_from_home = st.checkbox("Include from home (Urmston)", value=False)
    offer_curve_ball = st.checkbox("Offer curve ball", value=True)
    prox_override = st.checkbox("Let terrain & trail override proximity", value=True)
    st.caption("Tip: leave include-only empty to consider all locations.")

depart_dt = dt.datetime.combine(today, depart)
season_val = season_from_date(today) if season == "auto" else season

# Apply global overrides for scoring
set_tech_bias_override(tech_bias)
set_prox_override(prox_override)

# Location filtering
locs = [l for l in LOCATIONS if (not include_keys or l.key in include_keys) and (l.key not in exclude_keys)]
if not include_from_home:
    locs = [l for l in locs if l.key != "urmston"]

# ---------------- Scoring for Today / Tomorrow ----------------
rows_today, rows_tom = [], []
for loc in locs:
    try:
        r = score_location(loc, depart_dt, duration, terrain_bias, max_drive, season_val, tech_bias=tech_bias)
        rows_today.append(r)
    except Exception as e:
        rows_today.append({
            "key": loc.key, "name": loc.name, "score": 0.0,
            "components": {"weather":0,"trail":0,"proximity":0,"terrain_fit":0,"secondary":0},
            "drive_est_min": int(sum(loc.drive_min_typical)/2),
            "recommend_window": f"{depart_dt:%H:%M}–{(depart_dt + dt.timedelta(hours=duration)):%H:%M}",
            "notes": [f"Error: {e}"]
        })

depart_tom = depart_dt + dt.timedelta(days=1)
for loc in locs:
    try:
        r = score_location(loc, depart_tom, duration, terrain_bias, max_drive, season_val, tech_bias=tech_bias)
        rows_tom.append(r)
    except Exception as e:
        rows_tom.append({
            "key": loc.key, "name": loc.name, "score": 0.0,
            "components": {"weather":0,"trail":0,"proximity":0,"terrain_fit":0,"secondary":0},
            "drive_est_min": int(sum(loc.drive_min_typical)/2),
            "recommend_window": f"{depart_tom:%H:%M}–{(depart_tom + dt.timedelta(hours=duration)):%H:%M}",
            "notes": [f"Error: {e}"]
        })

rows_today = sorted(rows_today, key=lambda x: x["score"], reverse=True)
rows_tom = sorted(rows_tom, key=lambda x: x["score"], reverse=True)

# ---------------- Filter lists by Max drive ----------------
rows_today_ok = [r for r in rows_today if r.get("drive_est_min", 9999) <= max_drive]
rows_tom_ok = [r for r in rows_tom if r.get("drive_est_min", 9999) <= max_drive]

# Baseline: #1 filtered pick drive (fallback to best overall or max_drive)
base_today_drive = (rows_today_ok[0]['drive_est_min'] if rows_today_ok else (rows_today[0]['drive_est_min'] if rows_today else max_drive))
base_tom_drive = (rows_tom_ok[0]['drive_est_min'] if rows_tom_ok else (rows_tom[0]['drive_est_min'] if rows_tom else max_drive))
curve_extra_limit = 60  # minutes

# ---------------- Curve-ball logic ----------------
def pick_curve_ball(rows_all, rows_filtered, bias, tech_bias, baseline_drive_min: int, limit_extra_min: int):
    # Exclude the top-5 filtered picks; consider the remaining from the full set (which may include over-drive)
    top_keys = {r['key'] for r in rows_filtered[:5]}
    rest = [r for r in rows_all if r['key'] not in top_keys]
    # Constrain curve-ball drive to baseline + limit
    drive_cap = (baseline_drive_min or 0) + (limit_extra_min or 0)
    if drive_cap > 0:
        rest = [r for r in rest if r.get('drive_est_min', 9999) <= drive_cap]
    if not rest:
        return None

    min_wx = 70.0
    def want_sets(bias, tech_bias):
        want_elev = set()
        if bias > 0: want_elev = {"hills","steep","mod_elev"}
        elif bias < 0: want_elev = {"distance","flat","gravel"}
        want_tech = set()
        if tech_bias > 0: want_tech = {"technical","high_tech","steep"}
        elif tech_bias < 0: want_tech = {"gravel","flat","trail_centre","distance"}
        return want_elev, want_tech

    want_elev, want_tech = want_sets(bias, tech_bias)

    best = None; best_val = -1
    for r in rest:
        wx = r["components"]["weather"]; tags = terrain_map.get(r["key"], set())
        elev_ok = (not want_elev or (tags & want_elev))
        tech_ok = (not want_tech or (tags & want_tech))
        if wx >= min_wx and elev_ok and tech_ok:
            val = wx*1.0 + r["score"]*0.2
            if val > best_val:
                best = r; best_val = val

    if not best:
        best = max(rest, key=lambda x: x["components"]["weather"])
    return best

curve_today = pick_curve_ball(rows_today, rows_today_ok, terrain_bias, tech_bias, base_today_drive, curve_extra_limit)
curve_tom = pick_curve_ball(rows_tom, rows_tom_ok, terrain_bias, tech_bias, base_tom_drive, curve_extra_limit)

# ---------------- Map helpers ----------------
def build_features(rows, exclude_key=None):
    """Convert scored rows to ScatterplotLayer-friendly dicts, skipping exclude_key if provided."""
    feats = []
    for idx, r in enumerate(rows, start=1):
        if exclude_key and r['key'] == exclude_key:
            continue
        latlon = loc_lookup.get(r["key"])
        if not latlon:
            continue
        lat, lon = latlon
        # Size: overall score → radius meters
        radius_m = 800 + 45 * r["score"]
        # Rank colors
        if idx == 1:
            color = [0, 170, 0, 215]        # green
        elif idx == 2:
            color = [255, 200, 0, 215]      # yellow
        elif idx == 3:
            color = [30, 144, 255, 215]     # blue
        else:
            color = [128, 0, 128, 215]      # purple
        comps = r["components"]
        feats.append({
            "rank": idx,
            "name": r["name"],
            "score": r["score"],
            "lat": lat, "lon": lon,
            "radius": radius_m,
            "color": color,
            "weather": comps["weather"],
            "trail": comps["trail"],
            "proximity": comps["proximity"],
            "terrain_fit": comps["terrain_fit"],
            "secondary": comps["secondary"],
            "window": r["recommend_window"],
            "drive": r["drive_est_min"],
        })
    return feats

def meters_to_degrees(lat, dx_m, dy_m):
    # Convert meter offsets to lat/lon degree offsets at given latitude
    lat_rad = math.radians(lat)
    dlat = dy_m / 111320.0
    dlon = dx_m / (111320.0 * math.cos(lat_rad) if math.cos(lat_rad) != 0 else 1e-6)
    return dlat, dlon

def triangle_coords(lat, lon, size_m=2500):
    # Build an equilateral triangle centered at (lat, lon) with "radius" size_m
    # Points at 90°, 210°, 330° bearings
    angles = [90, 210, 330]
    coords = []
    for a in angles:
        rad = math.radians(a)
        dx = size_m * math.cos(rad)
        dy = size_m * math.sin(rad)
        dlat, dlon = meters_to_degrees(lat, dx, dy)
        coords.append([lon + dlon, lat + dlat])
    # Close the polygon
    coords.append(coords[0])
    return [coords]

# ---------------- Map rendering ----------------
st.divider()
st.subheader("Map of recommendations")
map_choice = st.radio("Show:", ["Today", "Tomorrow"], horizontal=True)
rows_for_map = rows_today_ok if map_choice == "Today" else rows_tom_ok
curve_for_map = curve_today if map_choice == "Today" else curve_tom

exclude_key = curve_for_map["key"] if (offer_curve_ball and curve_for_map) else None
features = build_features(rows_for_map, exclude_key=exclude_key)

if features or (offer_curve_ball and curve_for_map):
    # View centering
    all_lats = [f["lat"] for f in features]
    all_lons = [f["lon"] for f in features]
    if offer_curve_ball and curve_for_map:
        clat, clon = loc_lookup.get(curve_for_map["key"], (None, None))
        if clat is not None:
            all_lats.append(clat); all_lons.append(clon)
    avg_lat = sum(all_lats)/len(all_lats)
    avg_lon = sum(all_lons)/len(all_lons)

    layers = []

    if features:
        layer_points = pdk.Layer(
            "ScatterplotLayer",
            data=features,
            get_position='[lon, lat]',
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            stroked=True,
            get_line_color=[0, 0, 0],
            line_width_min_pixels=1,
        )
        layers.append(layer_points)

    # Curve-ball triangle (PolygonLayer only; no circle)
    if offer_curve_ball and curve_for_map is not None:
        clat, clon = loc_lookup.get(curve_for_map["key"], (None, None))
        if clat is not None:
            tri = triangle_coords(clat, clon, size_m=2500)  # larger triangle
            curve_poly = pdk.Layer(
                "PolygonLayer",
                data=[{
                    "rank": "CB",
                    "name": curve_for_map["name"],
                    "score": curve_for_map["score"],
                    "window": curve_for_map["recommend_window"],
                    "drive": curve_for_map["drive_est_min"],
                    "polygon": tri,
                }],
                get_polygon="polygon",
                get_fill_color=[255, 100, 0, 220],  # orange
                get_line_color=[0, 0, 0, 255],
                line_width_min_pixels=2,
                pickable=True,
            )
            layers.append(curve_poly)

    tooltip = {
        "html": (
            "<b>#{rank} {name}</b><br/>"
            "Score: <b>{score}</b><br/>"
            "Window: {window}<br/>"
            "Drive: ~{drive} min<br/>"
            "<hr style='margin:4px 0'/>"
            "Weather: {weather}<br/>"
            "Trail: {trail}<br/>"
            "Terrain fit: {terrain_fit}<br/>"
            "Proximity: {proximity}<br/>"
            "Secondary: {secondary}"
        ),
        "style": {"backgroundColor": "rgba(30,30,30,0.9)", "color": "white"}
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=7, pitch=0),
        map_style=None,
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

    # Legend
    with st.expander("Legend / colors"):
        st.markdown(
            "- **#1** green • **#2** yellow • **#3** blue • **#4+** purple\n\n"
            "- **Orange triangle** = curve ball (can be up to +60 min beyond #1 drive time)"
        )
else:
    st.info("No locations to display with current filters.")

st.divider()

# ---------------- Results tabs ----------------
tab1, tab2 = st.tabs(["Today", "Tomorrow"])

with tab1:
    st.subheader("Top picks — Today")
    for i, r in enumerate(rows_today_ok[:5], start=1):
        with st.container(border=True):
            st.markdown(f"### {i}) {r['name']} — **{r['score']}**")
            cols = st.columns(5); comps = r["components"]
            cols[0].metric("Weather", comps["weather"]); cols[1].metric("Trail", comps["trail"])
            cols[2].metric("Proximity", comps["proximity"]); cols[3].metric("Terrain fit", comps["terrain_fit"]); cols[4].metric("Secondary", comps["secondary"])
            st.write(f"**Window:** {r['recommend_window']} | **Drive:** ~{r['drive_est_min']} min")
            for n in r["notes"]:
                st.write(f"- {n}")
    st.subheader("Alternates — Today")
    for r in rows_today_ok[5:7]:
        st.write(f"- {r['name']} — {r['score']}")
    if offer_curve_ball and curve_today:
        st.subheader("Curve ball — Today")
        st.caption(f"Curve ball limited to ≤ {curve_extra_limit} min beyond today's #1 drive (~{base_today_drive}+{curve_extra_limit} min).")
        r = curve_today
        st.write(f"**{r['name']}** — {r['score']} (Weather {r['components']['weather']}, Terrain fit {r['components']['terrain_fit']})")
        for n in r["notes"][:3]:
            st.write(f"- {n}")

with tab2:
    st.subheader("Top picks — Tomorrow (same depart time)")
    for i, r in enumerate(rows_tom_ok[:5], start=1):
        with st.container(border=True):
            st.markdown(f"### {i}) {r['name']} — **{r['score']}**")
            cols = st.columns(5); comps = r["components"]
            cols[0].metric("Weather", comps["weather"]); cols[1].metric("Trail", comps["trail"])
            cols[2].metric("Proximity", comps["proximity"]); cols[3].metric("Terrain fit", comps["terrain_fit"]); cols[4].metric("Secondary", comps["secondary"])
            st.write(f"**Window:** {r['recommend_window']} | **Drive:** ~{r['drive_est_min']} min")
            for n in r["notes"]:
                st.write(f"- {n}")
    st.subheader("Alternates — Tomorrow")
    for r in rows_tom_ok[5:7]:
        st.write(f"- {r['name']} — {r['score']}")
    if offer_curve_ball and curve_tom:
        st.subheader("Curve ball — Tomorrow")
        st.caption(f"Curve ball limited to ≤ {curve_extra_limit} min beyond tomorrow's #1 drive (~{base_tom_drive}+{curve_extra_limit} min).")
        r = curve_tom
        st.write(f"**{r['name']}** — {r['score']} (Weather {r['components']['weather']}, Terrain fit {r['components']['terrain_fit']})")
        for n in r["notes"][:3]:
            st.write(f"- {n}")
