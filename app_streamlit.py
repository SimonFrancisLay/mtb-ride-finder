import streamlit as st
import datetime as dt
import math

# Try to import pydeck; if missing, show a warning but don't crash
try:
    import pydeck as pdk
    HAVE_PYDECK = True
except Exception as _e:
    HAVE_PYDECK = False

from mtb_agent import (
    LOCATIONS,
    season_from_date,
    score_location,
    set_tech_bias_override,
    set_prox_override,
    set_home_by_key,
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

# Apply settings
depart_dt = dt.datetime.combine(today, depart)
season_val = season_from_date(today) if season == "auto" else season

# Home / overrides
set_home_by_key(home_key, LOCATIONS)
set_tech_bias_override(tech_bias)
set_prox_override(prox_override)

# Filter set (no special-casing Urmston now)
locs = [l for l in LOCATIONS if (not include_keys or l.key in include_keys) and (l.key not in exclude_keys)]

# ---------------- Scoring ----------------
def score_all(locs, when_dt):
    rows = []
    for loc in locs:
        try:
            r = score_location(loc, when_dt, duration, terrain_bias, max_drive, season_val, tech_bias=tech_bias)
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

rows_today = score_all(locs, depart_dt)
rows_tomorrow = score_all(locs, depart_dt + dt.timedelta(days=1))

# Filter by Max drive for primary picks
def in_cap(rows): return [r for r in rows if r.get("drive_est_min", 9999) <= max_drive]
rows_today_ok = in_cap(rows_today)
rows_tomorrow_ok = in_cap(rows_tomorrow)

# Baselines for curve-ball (+60 rule)
def baseline_drive(filtered, allrows):
    if filtered:
        return filtered[0]['drive_est_min']
    if allrows:
        return allrows[0]['drive_est_min']
    return max_drive

base_today_drive = baseline_drive(rows_today_ok, rows_today)
base_tom_drive   = baseline_drive(rows_tomorrow_ok, rows_tomorrow)
curve_extra_limit = 60  # minutes

# Curve-ball selection
def pick_curve_ball(rows_all, rows_filtered, bias, tech_bias, baseline_drive_min: int, limit_extra_min: int):
    # Exclude the top-5 filtered picks; consider the remaining from the full set (may include over-drive)
    top_keys = {r['key'] for r in rows_filtered[:5]}
    rest = [r for r in rows_all if r['key'] not in top_keys]

    # Constrain curve-ball drive to baseline + limit
    drive_cap = (baseline_drive_min or 0) + (limit_extra_min or 0)
    rest = [r for r in rest if r.get('drive_est_min', 9999) <= drive_cap]

    if not rest:
        return None

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
            if val > best_val:
                best, best_val = r, val

    if not best:
        best = max(rest, key=lambda x: x["components"]["weather"])
    return best

curve_today = pick_curve_ball(rows_today, rows_today_ok, terrain_bias, tech_bias, base_today_drive, curve_extra_limit) if offer_curve_ball else None
curve_tomorrow = pick_curve_ball(rows_tomorrow, rows_tomorrow_ok, terrain_bias, tech_bias, base_tom_drive, curve_extra_limit) if offer_curve_ball else None

# ---------------- Map helpers ----------------
def build_features(rows, exclude_key=None):
    feats = []
    for idx, r in enumerate(rows, start=1):
        if exclude_key and r['key'] == exclude_key:
            continue
        latlon = loc_lookup.get(r["key"])
        if not latlon:
            continue
        lat, lon = latlon
        radius_m = 800 + 45 * r["score"]
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
    lat_rad = math.radians(lat)
    dlat = dy_m / 111320.0
    dlon = dx_m / (111320.0 * math.cos(lat_rad) if math.cos(lat_rad) != 0 else 1e-6)
    return dlat, dlon

def triangle_coords(lat, lon, size_m=2500):
    angles = [90, 210, 330]
    coords = []
    for a in angles:
        rad = math.radians(a)
        dx = size_m * math.cos(rad)
        dy = size_m * math.sin(rad)
        dlat, dlon = meters_to_degrees(lat, dx, dy)
        coords.append([lon + dlon, lat + dlat])
    coords.append(coords[0])
    return [coords]

# ---------------- Map rendering ----------------
st.divider()
st.subheader("Map of recommendations")
map_choice = st.radio("Show:", ["Today", "Tomorrow"], horizontal=True)
rows_for_map = rows_today_ok if map_choice == "Today" else rows_tomorrow_ok
curve_for_map = curve_today if map_choice == "Today" else curve_tomorrow

exclude_key = curve_for_map["key"] if (offer_curve_ball and curve_for_map) else None
features = build_features(rows_for_map, exclude_key=exclude_key)

if not HAVE_PYDECK:
    st.warning("pydeck is not installed, so the map is hidden.\n\nInstall with:\n\n```bash\npython -m pip install pydeck\n```")
elif features or (offer_curve_ball and curve_for_map):
    all_lats = [f["lat"] for f in features]
    all_lons = [f["lon"] for f in features]
    if offer_curve_ball and curve_for_map:
        clat, clon = loc_lookup.get(curve_for_map["key"], (None, None))
        if clat is not None:
            all_lats.append(clat); all_lons.append(clon)
    avg_lat = sum(all_lats)/len(all_lats) if all_lats else 53.48
    avg_lon = sum(all_lons)/len(all_lons) if all_lons else -2.3

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
    else:
        layer_points = None
    if layer_points is not None:
        layers.append(layer_points)

    # Curve-ball triangle (only triangle; no circle)
    if offer_curve_ball and curve_for_map is not None:
        clat, clon = loc_lookup.get(curve_for_map["key"], (None, None))
        if clat is not None:
            tri = triangle_coords(clat, clon, size_m=2500)
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
                get_fill_color=[255, 100, 0, 220],
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

    with st.expander("Legend / colors"):
        st.markdown(
            "- **#1** green • **#2** yellow • **#3** blue • **#4+** purple\n\n"
            "- **Orange triangle** = curve ball (≤ +60 min beyond the #1 filtered drive)"
        )
else:
    st.info("No locations to display with current filters.")

st.divider()

# ---------------- Results tabs ----------------
def render_list(rows_ok, label, base_drive, curve_pick):
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
    if offer_curve_ball and curve_pick:
        st.subheader(f"Curve ball — {label}")
        st.caption(f"Limited to ≤ {curve_extra_limit} min beyond {label.lower()}'s #1 drive (~{base_drive}+{curve_extra_limit} min).")
        r = curve_pick
        st.write(f"**{r['name']}** — {r['score']} (Weather {r['components']['weather']}, Terrain fit {r['components']['terrain_fit']})")
        for n in r["notes"][:3]:
            st.write(f"- {n}")

tab_today, tab_tomorrow = st.tabs(["Today", "Tomorrow"])

with tab_today:
    render_list(rows_today_ok, "Today", base_today_drive, curve_today)
with tab_tomorrow:
    render_list(rows_tomorrow_ok, "Tomorrow", base_tom_drive, curve_tomorrow)
