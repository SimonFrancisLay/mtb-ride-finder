# config_creator.py — interactive editor for MTB Ride Finder config.yaml
# Run:  python -m streamlit run config_creator.py
from __future__ import annotations

import io
import os
import copy
import math
import datetime as dt
from typing import List, Dict, Any

import yaml
import pandas as pd
import streamlit as st

# ----------------------------- App config ------------------------------------
st.set_page_config(page_title="MTB Config Creator", layout="wide")

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

SCHEMA_HELP = {
    "key": "Unique slug (no spaces). Used as identifier.",
    "region": "Grouping used by the app (e.g., 'NW England', 'Yorkshire Dales').",
    "name": "Human-friendly location name.",
    "lat": "Latitude (decimal).",
    "lon": "Longitude (decimal).",
    "drive_min_typical": "Typical drive time [min_low, min_high]. If you leave [0,0], the app estimates from 'Home'.",
    "terrain": "Comma-separated tags (e.g., 'hills, steep, technical' or 'distance, flat, gravel').",
    "drainage": "One of: rocky, trail_centre, mixed, mixed_long_mud, peaty.",
    "mud_sensitivity": "One of: low, med, high.",
    "duration_range": "Recommended ride duration [min_hours, max_hours].",
    "notes": "Optional free text.",
}

RAIN_RESPONSE_FACTORS = {
    # These mirror the logic in mtb_agent.py (legacy recovery weights)
    # Higher = faster recovery / better in wet
    "drainage": {
        "trail_centre": 1.05,
        "rocky": 1.05,
        "mixed": 1.00,
        "mixed_long_mud": 0.90,
        "peaty": 0.90,
    },
    "mud_sensitivity": {
        "low": 1.05,
        "med": 1.00,
        "high": 0.92,
    },
}

# ----------------------------- Utilities -------------------------------------
def load_yaml(text: str) -> Dict[str, Any]:
    data = yaml.safe_load(text) or {}
    if "weights" not in data:
        data["weights"] = {
            "weather": 0.25,
            "trail": 0.35,
            "proximity": 0.10,
            "terrain_fit": 0.25,
            "secondary": 0.05,
        }
    if "locations" not in data:
        data["locations"] = []
    return data

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"weights": {"weather":0.25,"trail":0.35,"proximity":0.10,"terrain_fit":0.25,"secondary":0.05}, "locations": []}
    with open(path, "r", encoding="utf-8") as f:
        return load_yaml(f.read())

def dump_config(cfg: Dict[str, Any]) -> str:
    # Safe & pretty dump
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=100)

def ensure_schema(loc: Dict[str, Any]) -> Dict[str, Any]:
    # Fill required fields with sensible defaults
    out = {
        "key": loc.get("key","").strip(),
        "region": loc.get("region","").strip(),
        "name": loc.get("name","").strip(),
        "lat": float(loc.get("lat", 0.0) or 0.0),
        "lon": float(loc.get("lon", 0.0) or 0.0),
        "drive_min_typical": list(loc.get("drive_min_typical", [0,0]))[:2],
        "terrain": loc.get("terrain","").strip(),
        "drainage": loc.get("drainage","mixed").strip(),
        "mud_sensitivity": loc.get("mud_sensitivity","med").strip(),
        "duration_range": list(loc.get("duration_range", [2.0,4.0]))[:2],
        "notes": loc.get("notes",""),
    }
    # normalize arrays
    if len(out["drive_min_typical"]) < 2:
        out["drive_min_typical"] = [0,0]
    if len(out["duration_range"]) < 2:
        out["duration_range"] = [2.0,4.0]
    return out

def validate_locations(locs: List[Dict[str, Any]]) -> List[str]:
    errors = []
    keys = set()
    for i, l in enumerate(locs):
        k = l.get("key","").strip()
        if not k:
            errors.append(f"[#{i+1}] key is required.")
        elif " " in k:
            errors.append(f"[#{i+1}] key '{k}' must not contain spaces.")
        elif k in keys:
            errors.append(f"[#{i+1}] key '{k}' is duplicated.")
        keys.add(k)
        try:
            lat = float(l.get("lat", 0)); lon = float(l.get("lon", 0))
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                errors.append(f"[#{i+1}] '{k}' lat/lon out of bounds.")
        except Exception:
            errors.append(f"[#{i+1}] '{k}' lat/lon not numeric.")
        dr = l.get("drainage","mixed")
        if dr not in RAIN_RESPONSE_FACTORS["drainage"]:
            errors.append(f"[#{i+1}] '{k}' drainage '{dr}' not in {list(RAIN_RESPONSE_FACTORS['drainage'].keys())}.")
        ms = l.get("mud_sensitivity","med")
        if ms not in RAIN_RESPONSE_FACTORS["mud_sensitivity"]:
            errors.append(f"[#{i+1}] '{k}' mud_sensitivity '{ms}' not in {list(RAIN_RESPONSE_FACTORS['mud_sensitivity'].keys())}.")
        # drive_min_typical / duration_range shapes
        dmt = l.get("drive_min_typical",[0,0])
        if not (isinstance(dmt, list) and len(dmt) == 2):
            errors.append(f"[#{i+1}] '{k}' drive_min_typical must be [min,max].")
        dur = l.get("duration_range",[2.0,4.0])
        if not (isinstance(dur, list) and len(dur) == 2):
            errors.append(f"[#{i+1}] '{k}' duration_range must be [min,max].")
    return errors

def rain_response_index(dr: str, ms: str) -> float:
    # Convenience index to compare "how well it copes with wet"
    df = RAIN_RESPONSE_FACTORS["drainage"].get(dr, 1.0)
    mf = RAIN_RESPONSE_FACTORS["mud_sensitivity"].get(ms, 1.0)
    # scale to 0–100 just for visual comparison
    return round((df * mf) / (1.05 * 1.05) * 100.0, 1)  # best combo ~100

# ----------------------------- Sidebar I/O -----------------------------------
st.sidebar.subheader("Config file")

def is_writable_dir(path: str) -> bool:
    try:
        d = os.path.dirname(os.path.abspath(path)) or os.getcwd()
        return os.path.isdir(d) and os.access(d, os.W_OK)
    except Exception:
        return False

cfg_path = st.sidebar.text_input("Path to config.yaml", value=DEFAULT_CONFIG_PATH)
resolved = os.path.abspath(cfg_path)
st.sidebar.caption(f"Resolved path: `{resolved}`")

writable = is_writable_dir(resolved)
st.sidebar.caption("Directory is writable ✅" if writable else "Directory is not writable ❌ (download instead)")

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("Load"):
    st.session_state["cfg_raw"] = load_config(resolved)
    st.sidebar.success(f"Loaded from {resolved}")

# Overwrite and backup options
overwrite_ok = st.sidebar.checkbox("Allow overwrite existing file", value=True)
backup_on_overwrite = st.sidebar.checkbox("Create timestamped backup when overwriting", value=True)

if col_btn2.button("Save to disk"):
    cfg = st.session_state.get("cfg_raw") or {"weights": {}, "locations": []}
    text = dump_config(cfg)

    if not writable:
        st.sidebar.error("Target directory is not writable here. Use the Download button below and replace the file manually.")
    else:
        try:
            if os.path.exists(resolved):
                if not overwrite_ok:
                    st.sidebar.error("File exists and overwrite is disabled. Enable 'Allow overwrite existing file' or change the filename.")
                else:
                    # make a backup if requested
                    if backup_on_overwrite:
                        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                        backup_path = resolved.replace(".yaml", f".backup-{ts}.yaml")
                        try:
                            with open(backup_path, "w", encoding="utf-8") as bf:
                                bf.write(open(resolved, "r", encoding="utf-8").read())
                            st.sidebar.info(f"Backup created: {backup_path}")
                        except Exception as e:
                            st.sidebar.warning(f"Backup failed: {e}")
                    with open(resolved, "w", encoding="utf-8") as f:
                        f.write(text)
                    st.sidebar.success(f"Overwrote {resolved}")
            else:
                # fresh write
                with open(resolved, "w", encoding="utf-8") as f:
                    f.write(text)
                st.sidebar.success(f"Saved new file at {resolved}")
        except Exception as e:
            st.sidebar.error(f"Failed to save: {e}")

st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("Upload config.yaml", type=["yaml","yml"], help="Load a YAML to edit.")
if uploaded is not None:
    text = uploaded.read().decode("utf-8")
    st.session_state["cfg_raw"] = load_yaml(text)
    st.sidebar.success("Loaded from uploaded file.")

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload config.yaml", type=["yaml","yml"], help="Load a YAML to edit.")
if uploaded is not None:
    text = uploaded.read().decode("utf-8")
    st.session_state["cfg_raw"] = load_yaml(text)

if "cfg_raw" not in st.session_state:
    st.session_state["cfg_raw"] = load_config(cfg_path)

cfg = st.session_state["cfg_raw"]

# ----------------------------- Weights editor --------------------------------
st.header("Weights (used by the main app)")
w = cfg.get("weights", {})
c = st.columns(5)
w["weather"] = c[0].number_input("weather", 0.0, 1.0, float(w.get("weather", 0.25)), 0.01, help="Primary weather driver.")
w["trail"] = c[1].number_input("trail", 0.0, 1.0, float(w.get("trail", 0.35)), 0.01, help="Legacy trail condition blend.")
w["proximity"] = c[2].number_input("proximity", 0.0, 1.0, float(w.get("proximity", 0.10)), 0.01, help="Travel time impact.")
w["terrain_fit"] = c[3].number_input("terrain_fit", 0.0, 1.0, float(w.get("terrain_fit", 0.25)), 0.01, help="Distance vs hills + chilled/gnar.")
w["secondary"] = c[4].number_input("secondary", 0.0, 1.0, float(w.get("secondary", 0.05)), 0.01, help="Tie-breaker/sun/temp etc.")
# Normalize preview
total = max(1e-6, sum(w.values()))
norm = {k: round(v/total, 3) for k,v in w.items()}
st.caption(f"Normalised preview → {norm}")

cfg["weights"] = w

st.markdown("---")

# ----------------------------- Locations browser -----------------------------
st.header("Locations")

# Filters & compare options
regions = sorted({loc.get("region","") for loc in cfg.get("locations", []) if loc.get("region")})
colf1, colf2, colf3 = st.columns([2,2,2])
sel_regions = colf1.multiselect("Filter region(s)", options=regions)
txt_search = colf2.text_input("Name / key contains", "")
sort_by = colf3.selectbox("Sort by", ["name","region","key","rain_response","duration_min","duration_max"])

def as_row(l: Dict[str, Any]) -> Dict[str, Any]:
    l2 = ensure_schema(l)
    rri = rain_response_index(l2["drainage"], l2["mud_sensitivity"])
    return {
        "key": l2["key"],
        "region": l2["region"],
        "name": l2["name"],
        "lat": l2["lat"],
        "lon": l2["lon"],
        "drive_min_low": l2["drive_min_typical"][0],
        "drive_min_high": l2["drive_min_typical"][1],
        "terrain": l2["terrain"],
        "drainage": l2["drainage"],
        "mud_sensitivity": l2["mud_sensitivity"],
        "rain_response": rri,
        "duration_min": l2["duration_range"][0],
        "duration_max": l2["duration_range"][1],
        "notes": l2["notes"],
    }

rows = [as_row(l) for l in cfg.get("locations", [])]
df = pd.DataFrame(rows)

# Apply filters
if sel_regions:
    df = df[df["region"].isin(sel_regions)]
if txt_search.strip():
    s = txt_search.strip().lower()
    df = df[df.apply(lambda r: s in str(r["name"]).lower() or s in str(r["key"]).lower(), axis=1)]

# Sort
if sort_by == "rain_response":
    df = df.sort_values(by="rain_response", ascending=False)
elif sort_by in ("duration_min","duration_max"):
    df = df.sort_values(by=sort_by, ascending=False)
else:
    df = df.sort_values(by=sort_by)

st.subheader("Compare side-by-side")
st.dataframe(
    df[[
        "region","name","key","lat","lon",
        "drive_min_low","drive_min_high",
        "terrain","drainage","mud_sensitivity","rain_response",
        "duration_min","duration_max","notes"
    ]],
    use_container_width=True,
)

st.caption("**Rain response** is a quick index of how well a location tends to recover/drain after rain, derived from drainage & mud sensitivity (higher = better).")

# ----------------------------- Edit / Add ------------------------------------
st.markdown("### Edit / Add a location")

# pick target
existing_keys = [r["key"] for _, r in df.iterrows()]
sel_key = st.selectbox("Select existing (or pick blank to add new)", ["<new>"] + existing_keys, index=0)

if sel_key != "<new>":
    # load selected into form defaults
    selected = next((ensure_schema(l) for l in cfg["locations"] if l.get("key") == sel_key), ensure_schema({}))
else:
    selected = ensure_schema({})

with st.form("loc_form", clear_on_submit=False):
    c1, c2, c3 = st.columns([1.2,1,1])
    key = c1.text_input("key", value=selected["key"], help=SCHEMA_HELP["key"])
    region = c2.text_input("region", value=selected["region"], help=SCHEMA_HELP["region"])
    name = c3.text_input("name", value=selected["name"], help=SCHEMA_HELP["name"])

    c4, c5, c6 = st.columns(3)
    lat = c4.number_input("lat", value=float(selected["lat"]), step=0.0005, format="%.6f", help=SCHEMA_HELP["lat"])
    lon = c5.number_input("lon", value=float(selected["lon"]), step=0.0005, format="%.6f", help=SCHEMA_HELP["lon"])
    drive_low = c6.number_input("drive_min_low", value=int(selected["drive_min_typical"][0]), step=5, help=SCHEMA_HELP["drive_min_typical"])

    c7, c8, c9 = st.columns(3)
    drive_high = c7.number_input("drive_min_high", value=int(selected["drive_min_typical"][1]), step=5)
    terrain = c8.text_input("terrain", value=selected["terrain"], help=SCHEMA_HELP["terrain"])
    drainage = c9.selectbox("drainage", options=list(RAIN_RESPONSE_FACTORS["drainage"].keys()), index=list(RAIN_RESPONSE_FACTORS["drainage"].keys()).index(selected["drainage"] if selected["drainage"] in RAIN_RESPONSE_FACTORS["drainage"] else "mixed"))

    c10, c11, c12 = st.columns(3)
    mud_sensitivity = c10.selectbox("mud_sensitivity", options=list(RAIN_RESPONSE_FACTORS["mud_sensitivity"].keys()), index=list(RAIN_RESPONSE_FACTORS["mud_sensitivity"].keys()).index(selected["mud_sensitivity"] if selected["mud_sensitivity"] in RAIN_RESPONSE_FACTORS["mud_sensitivity"] else "med"))
    duration_min = c11.number_input("duration_min", value=float(selected["duration_range"][0]), step=0.5, help=SCHEMA_HELP["duration_range"])
    duration_max = c12.number_input("duration_max", value=float(selected["duration_range"][1]), step=0.5)

    notes = st.text_area("notes", value=selected["notes"], help=SCHEMA_HELP["notes"])

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    submitted = fcol1.form_submit_button("Save/Update")
    dup_clicked = fcol2.form_submit_button("Duplicate as new")
    del_clicked = fcol3.form_submit_button("Delete")
    clear_clicked = fcol4.form_submit_button("Clear form")

    if submitted or dup_clicked:
        new_entry = ensure_schema({
            "key": key if not dup_clicked else f"{key}_copy",
            "region": region,
            "name": name,
            "lat": lat,
            "lon": lon,
            "drive_min_typical": [int(drive_low), int(drive_high)],
            "terrain": terrain,
            "drainage": drainage,
            "mud_sensitivity": mud_sensitivity,
            "duration_range": [float(duration_min), float(duration_max)],
            "notes": notes,
        })

        locs = [ensure_schema(l) for l in cfg["locations"]]
        if dup_clicked or key not in [l["key"] for l in locs]:
            # add new
            locs.append(new_entry)
        else:
            # update existing by key
            for i, l in enumerate(locs):
                if l["key"] == key:
                    locs[i] = new_entry
                    break

        errors = validate_locations(locs)
        if errors:
            st.error("Validation errors:\n- " + "\n- ".join(errors))
        else:
            cfg["locations"] = locs
            st.success("Saved to in-memory config (use 'Save to disk' in the sidebar to write file).")

    if del_clicked and sel_key != "<new>":
        before = len(cfg["locations"])
        cfg["locations"] = [l for l in cfg["locations"] if l.get("key") != sel_key]
        after = len(cfg["locations"])
        if after < before:
            st.warning(f"Deleted '{sel_key}' from in-memory config.")
        else:
            st.info("Nothing deleted.")

    if clear_clicked:
        st.experimental_rerun()

st.markdown("---")

# ----------------------------- Export section --------------------------------
st.header("Export")
colE1, colE2 = st.columns(2)

# YAML export
yaml_text = dump_config(cfg)
colE1.download_button(
    label="⬇️ Download config.yaml",
    data=yaml_text.encode("utf-8"),
    file_name="config.yaml",
    mime="text/yaml",
    help="Saves weights + locations exactly as shown here."
)

# CSV export (locations only, convenient for spreadsheet edits)
loc_csv = pd.DataFrame([ensure_schema(l) for l in cfg["locations"]])
colE2.download_button(
    label="⬇️ Download locations.csv",
    data=loc_csv.to_csv(index=False).encode("utf-8"),
    file_name="locations.csv",
    mime="text/csv",
    help="Only location rows. You can re-import manually via YAML."
)

with st.expander("Raw YAML (preview)"):
    st.code(yaml_text, language="yaml")

st.caption("Tip: keep this editor and your ride app open in two tabs. Edit here → download → refresh your ride app.")