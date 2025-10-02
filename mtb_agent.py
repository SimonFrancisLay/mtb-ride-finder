#!/usr/bin/env python3
import argparse
import math
import requests
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import statistics as stats
import yaml
import os
import time

LONDON_TZ = "Europe/London"
TECH_BIAS_OVERRIDE = None
PROX_OVERRIDE = True

# --- Hourly caches (TTL ~ 3600s) ---
_WEATHER_CACHE: Dict[Tuple[float, float, str], Tuple[float, dict]] = {}
_DRIVE_CACHE: Dict[Tuple[str, str, int], Tuple[float, int]] = {}

@dataclass
class Location:
    key: str
    name: str
    lat: float
    lon: float
    drive_min_typical: Tuple[int, int]
    terrain: str
    drainage: str
    mud_sensitivity: str
    duration_range: Tuple[float, float]
    notes: str

# ---------- YAML loading ----------
def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _location_from_dict(d: dict) -> Location:
    return Location(
        key=d["key"],
        name=d["name"],
        lat=float(d["lat"]),
        lon=float(d["lon"]),
        drive_min_typical=(int(d["drive_min_typical"][0]), int(d["drive_min_typical"][1])),
        terrain=d["terrain"],
        drainage=d["drainage"],
        mud_sensitivity=d.get("mud_sensitivity", "med"),
        duration_range=(float(d["duration_range"][0]), float(d["duration_range"][1])),
        notes=d.get("notes", ""),
    )

def load_locations_from_config(cfg: dict) -> List[Location]:
    locs = []
    for d in cfg.get("locations", []):
        try:
            locs.append(_location_from_dict(d))
        except Exception:
            # Skip malformed entries without crashing
            pass
    return locs

# ---------- Season / util ----------
def season_from_date(d: dt.date) -> str:
    m = d.month
    if m in (12, 1, 2): return "winter"
    if m in (3, 4, 5): return "spring"
    if m in (6, 7, 8): return "summer"
    return "autumn"

def parse_time_str(tstr: str) -> Tuple[int, int]:
    hh, mm = tstr.split(":"); return int(hh), int(mm)

def rush_hour_multiplier(depart_hh: int) -> float:
    return 1.15 if (7 <= depart_hh <= 9 or 16 <= depart_hh <= 18) else 1.0

def clamp(v, lo, hi): return max(lo, min(hi, v))

def map_range(x, a, b, c, d):
    if b == a: return (c + d) / 2
    t = (x - a) / (b - a)
    t = clamp(t, 0, 1)
    return c + t * (d - c)

# ---------- Weather (hourly cached) ----------
def _weather_cache_key(lat: float, lon: float, timezone: str) -> Tuple[float, float, str]:
    rl = round(lat, 2); rlon = round(lon, 2)
    hour_key = dt.datetime.now().strftime("%Y%m%d%H") + timezone
    return (rl, rlon, hour_key)

def fetch_open_meteo(lat: float, lon: float, timezone: str, past_days: int = 7) -> dict:
    key = _weather_cache_key(lat, lon, timezone)
    now = time.time()
    cached = _WEATHER_CACHE.get(key)
    if cached and (now - cached[0] < 3600):
        return cached[1]

    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m,cloudcover,temperature_2m",
        "daily": "precipitation_sum,sunshine_duration,temperature_2m_max,temperature_2m_min",
        "forecast_days": 2, "past_days": past_days, "timezone": timezone
    }
    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    _WEATHER_CACHE[key] = (now, data)
    return data

# ---------- Scoring ----------
def score_weather(hourly: dict, depart: dt.datetime, duration_h: float) -> float:
    times = [dt.datetime.fromisoformat(t) for t in hourly["time"]]
    end = depart + dt.timedelta(hours=duration_h)
    idx = [i for i, t in enumerate(times) if depart <= t < end]
    if not idx: return 50.0

    def sel(k): return [hourly[k][i] for i in idx]
    avg_prob = stats.mean(sel("precipitation_probability"))
    avg_mm = stats.mean(sel("precipitation"))
    avg_w = stats.mean(sel("wind_speed_10m")); avg_g = stats.mean(sel("wind_gusts_10m"))
    avg_cloud = stats.mean(sel("cloudcover")); avg_temp = stats.mean(sel("temperature_2m"))

    rain_penalty = clamp(avg_prob * (1 + avg_mm), 0, 100)
    rain_score = 100 - map_range(rain_penalty, 0, 100, 0, 60)
    wind_index = avg_w * 0.6 + avg_g * 0.4
    wind_score = 100 - map_range(wind_index, 0, 50, 0, 50)
    sun_score = map_range(100 - avg_cloud, 0, 100, 40, 100)
    cold_pen = map_range(10 - avg_temp, 0, 15, 0, 25) if avg_temp < 10 else 0
    heat_pen = map_range(avg_temp - 22, 0, 12, 0, 25) if avg_temp > 22 else 0
    temp_score = 100 - clamp(cold_pen + heat_pen, 0, 40)
    return clamp(0.5 * rain_score + 0.3 * wind_score + 0.1 * sun_score + 0.1 * temp_score, 0, 100)

def trail_dryness_score(drainage: str, season: str, precip_7d_mm: float) -> float:
    season_mult = {"winter": 0.8, "spring": 0.95, "summer": 1.1, "autumn": 0.9}.get(season, 1.0)
    adj_mm = precip_7d_mm / max(season_mult, 0.1)
    curves = {"trail_centre": 0.030, "rocky": 0.035, "mixed": 0.045, "mixed_long_mud": 0.055, "moor_slow": 0.065}
    k = curves.get(drainage, 0.045)
    return clamp(100 * math.exp(-k * adj_mm), 0, 100)

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# ----- Home selection & drive estimation (hourly cached) -----
HOME_KEY: Optional[str] = None
HOME_COORDS: Optional[Tuple[float, float]] = None

def set_home_by_key(key: Optional[str], locs: List[Location]) -> None:
    """Set the 'start/home' by location key. None = fallback to Urmston (or first)."""
    global HOME_KEY, HOME_COORDS
    HOME_KEY = key
    HOME_COORDS = None
    if key:
        for l in locs:
            if l.key == key:
                HOME_COORDS = (l.lat, l.lon)
                return
    # fallback to Urmston
    for l in locs:
        if l.key == "urmston":
            HOME_COORDS = (l.lat, l.lon)
            HOME_KEY = "urmston"
            return
    # final fallback: first
    if locs:
        HOME_COORDS = (locs[0].lat, locs[0].lon)
        HOME_KEY = locs[0].key

def drive_minutes_from_home(home_key: str, home_lat: float, home_lon: float, loc: Location, depart_hh: int) -> int:
    """Estimate drive time from chosen home using straight-line distance (haversine) -> avg road speed, cached hourly."""
    now = time.time()
    k = (home_key or "custom", loc.key, int(depart_hh))
    cached = _DRIVE_CACHE.get(k)
    if cached and (now - cached[0] < 3600):
        return cached[1]

    dist_km = _haversine_km(home_lat, home_lon, loc.lat, loc.lon)
    # Speed model: 65 km/h baseline (mixed A-roads/motorway); clamp to reasonable bounds
    mins = int((dist_km / 65.0) * 60.0)
    mins = int(mins * rush_hour_multiplier(depart_hh))
    mins = max(8, min(600, mins))

    _DRIVE_CACHE[k] = (now, mins)
    return mins

def score_proximity(est_drive_min: int, max_drive: int) -> float:
    if est_drive_min > max_drive: return 0.0
    return map_range(est_drive_min, 0, max_drive, 100, 0)

def score_terrain_fit(terrain_tags: str, pref_bias: float, tech_bias: float, duration_h: float, loc_range: Tuple[float, float]) -> float:
    tags = set([t.strip() for t in terrain_tags.split(",")])
    has_hills = any(t in tags for t in ["hills", "steep", "mod_elev"])
    has_distance = any(t in tags for t in ["distance", "flat", "gravel"])
    has_gnar = any(t in tags for t in ["technical", "high_tech", "trail_centre", "steep"])
    has_chilled = any(t in tags for t in ["gravel", "flat", "distance"])

    score = 30.0
    b = clamp(pref_bias, -1.0, 1.0)
    if b > 0:  # hills wanted
        score += 25.0 * (1.0 if has_hills else 0.0) * abs(b) - 8.0 * (0.0 if has_hills else 1.0) * abs(b)
    elif b < 0:  # distance wanted
        score += 25.0 * (1.0 if has_distance else 0.0) * abs(b) - 8.0 * (0.0 if has_distance else 1.0) * abs(b)

    tb = clamp(tech_bias, -1.0, 1.0)
    if tb > 0:  # gnar
        score += 25.0 * (1.0 if has_gnar else 0.0) * abs(tb) - 8.0 * (0.0 if has_gnar else 1.0) * abs(tb)
    elif tb < 0:  # chilled
        score += 25.0 * (1.0 if has_chilled else 0.0) * abs(tb) - 8.0 * (0.0 if has_chilled else 1.0) * abs(tb)

    lo, hi = loc_range
    if lo - 0.25 <= duration_h <= hi + 0.25:
        score += 20
    elif duration_h < lo:
        score -= map_range(lo - duration_h, 0, 2, 0, 20)
    else:
        score -= map_range(duration_h - hi, 0, 2, 0, 20)

    return clamp(score, 0, 100)

def score_secondary(hourly: dict, depart: dt.datetime, duration_h: float) -> float:
    times = [dt.datetime.fromisoformat(t) for t in hourly["time"]]
    end = depart + dt.timedelta(hours=duration_h)
    idx = [i for i, t in enumerate(times) if depart <= t < end]
    if not idx: return 50.0

    def sel(key): return [hourly[key][i] for i in idx]
    temps = sel("temperature_2m"); clouds = sel("cloudcover")
    cloudiness = sum(clouds) / len(clouds)
    cloud_pen = map_range(cloudiness, 0, 100, 0, 15)
    tmin, tmax = min(temps), max(temps)
    cold_pen = map_range(5 - tmin, 0, 10, 0, 25) if tmin < 5 else 0
    heat_pen = map_range(tmax - 24, 0, 12, 0, 25) if tmax > 24 else 0
    secondary = 100 - (cloud_pen + cold_pen + heat_pen)
    return clamp(secondary, 0, 100)

def assemble_reason(loc: Location, wx_s: float, tr_s: float, prox_s: float, terr_s: float, sec_s: float,
                    depart: dt.datetime, duration_h: float, drive_est: int, precip7: float,
                    wind_mean: float, gust_mean: float, rain_prob: float) -> List[str]:
    notes = []
    notes.append(f"Dryness: 7-day rain {precip7:.0f} mm → trail {int(tr_s)}/100.")
    notes.append(f"Wind: {int(wind_mean)} mph, gust {int(gust_mean)} mph; Weather score {int(wx_s)}.")
    notes.append(f"Rain probability in window: {int(rain_prob)}%.")
    if drive_est > 0: notes.append(f"Drive: ~{drive_est} min at {depart:%H:%M}. Proximity {int(prox_s)}.")
    else: notes.append("Start from home; no drive needed.")
    if "trail centre" in loc.notes.lower() or "trail_centre" in loc.terrain:
        notes.append("Trail-centre surfaces stay rideable even when wet.")
    return notes

def set_tech_bias_override(v: float):
    global TECH_BIAS_OVERRIDE; TECH_BIAS_OVERRIDE = v

def set_prox_override(v: bool):
    global PROX_OVERRIDE; PROX_OVERRIDE = bool(v)

def score_location(
    loc: Location,
    depart_dt: dt.datetime,
    duration_h: float,
    terrain_bias: float,
    max_drive: int,
    season: str,
    tech_bias: float = None,
    home_key: Optional[str] = None,
    home_coords: Optional[Tuple[float, float]] = None,
):
    # Weather
    data = fetch_open_meteo(loc.lat, loc.lon, LONDON_TZ, past_days=7)
    hourly = data["hourly"]; daily = data["daily"]
    wx_s = score_weather(hourly, depart_dt, duration_h)

    precip_list = daily.get("precipitation_sum", [])
    precip7 = sum(precip_list[-7:]) if len(precip_list) >= 7 else sum(precip_list)
    tr_s = trail_dryness_score(loc.drainage, season, precip7)

    # Drive estimation from chosen 'home'
    hk = None; hcoords = None
    if home_key is not None or home_coords is not None:
        hk = home_key or "custom"
        hcoords = home_coords
    elif HOME_COORDS is not None:
        hk, hcoords = (HOME_KEY or "custom"), HOME_COORDS

    if hcoords is not None:
        drive_est = drive_minutes_from_home(hk, hcoords[0], hcoords[1], loc, depart_dt.hour)
    else:
        drive_mid = sum(loc.drive_min_typical) / 2
        drive_est = int(drive_mid * rush_hour_multiplier(depart_dt.hour))

    prox_s = score_proximity(drive_est, max_drive)
    tb = TECH_BIAS_OVERRIDE if TECH_BIAS_OVERRIDE is not None else (tech_bias or 0.0)
    terr_s = score_terrain_fit(loc.terrain, terrain_bias, tb, duration_h, loc.duration_range)
    sec_s = score_secondary(hourly, depart_dt, duration_h)

    # Weights (Terrain & Trail emphasized)
    weights = {"weather": 0.25, "trail": 0.35, "proximity": 0.10, "terrain_fit": 0.25, "secondary": 0.05}

    # Terrain+Trail override proximity floor if enabled
    if PROX_OVERRIDE and terr_s >= 70 and tr_s >= 70:
        prox_s = max(prox_s, 50)

    total = (weights["weather"] * wx_s + weights["trail"] * tr_s + weights["proximity"] * prox_s +
             weights["terrain_fit"] * terr_s + weights["secondary"] * sec_s)

    # Window stats for notes
    times = [dt.datetime.fromisoformat(t) for t in hourly["time"]]
    end = depart_dt + dt.timedelta(hours=duration_h)
    idx = [i for i, t in enumerate(times) if depart_dt <= t < end]
    if idx:
        wind_mean = sum(hourly["wind_speed_10m"][i] for i in idx) / len(idx)
        gust_mean = sum(hourly["wind_gusts_10m"][i] for i in idx) / len(idx)
        rain_prob = sum(hourly["precipitation_probability"][i] for i in idx) / len(idx)
    else:
        wind_mean = gust_mean = rain_prob = 0.0

    reason = assemble_reason(loc, wx_s, tr_s, prox_s, terr_s, sec_s, depart_dt, duration_h,
                             drive_est, precip7, wind_mean, gust_mean, rain_prob)

    return {
        "key": loc.key, "name": loc.name, "score": round(total, 1),
        "components": {
            "weather": round(wx_s, 1), "trail": round(tr_s, 1), "proximity": round(prox_s, 1),
            "terrain_fit": round(terr_s, 1), "secondary": round(sec_s, 1)
        },
        "drive_est_min": int(drive_est),
        "recommend_window": f"{depart_dt:%H:%M}–{(depart_dt + dt.timedelta(hours=duration_h)):%H:%M}",
        "notes": reason
    }

def render_output(results: List[Dict], top_n: int = 5):
    print("\nDaily MTB locations — ranked\n")
    for i, r in enumerate(sorted(results, key=lambda x: x["score"], reverse=True)[:top_n], start=1):
        print(f"{i}) {r['name']} — {r['score']}")
        comps = r["components"]
        print(f"   • Wx {comps['weather']} | Trail {comps['trail']} | Prox {comps['proximity']} | Terrain {comps['terrain_fit']} | Secondary {comps['secondary']}")
        print(f"   • Window: {r['recommend_window']} | Drive ≈ {r['drive_est_min']} min")
        for n in r["notes"][:3]:
            print(f"   • {n}")
        if len(r["notes"]) > 3:
            print("   • …")
        print()

def set_home_default_from_cfg(locs: List[Location], cfg: dict):
    # Called at import to set home to Urmston by default
    home_label = cfg.get("start_location", "Urmston")
    # Prefer exact key match first
    for l in locs:
        if l.key == "urmston":
            set_home_by_key("urmston", locs)
            return
    # Fallback: by name contains
    for l in locs:
        if home_label.lower() in l.name.lower():
            set_home_by_key(l.key, locs)
            return
    # Fallback: first entry
    if locs:
        set_home_by_key(locs[0].key, locs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depart", type=str, default=None, help='Depart time "HH:MM" local')
    parser.add_argument("--max-drive", type=int, default=None, help="Max drive minutes")
    parser.add_argument("--terrain-bias", type=float, default=None, help="Continuous hills(+1) ↔ distance(-1)")
    parser.add_argument("--tech-bias", type=float, default=None, help="Continuous gnar(+1) ↔ chilled(-1)")
    parser.add_argument("--duration", type=float, default=None, help="Planned ride hours")
    parser.add_argument("--home", type=str, default=None, help="Home/start location key (e.g., 'urmston')")
    parser.add_argument("--season", type=str, default=None, choices=["winter", "spring", "summer", "autumn"])
    parser.add_argument("--config", type=str, default="config.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    locs = load_locations_from_config(cfg)
    if not locs:
        raise RuntimeError("No locations found in config.yaml under 'locations'.")

    # Set defaults
    depart_str = args.depart or cfg.get("depart_time", "08:00")
    max_drive = args.max_drive or int(cfg.get("max_drive_min", 90))
    terrain_bias = args.terrain_bias if args.terrain_bias is not None else float(cfg.get("preference", {}).get("terrain_bias", 0.0))
    tech_bias = args.tech_bias if args.tech_bias is not None else float(cfg.get("preference", {}).get("terrain_tech_bias", 0.0))
    duration_h = args.duration or float(cfg.get("preference", {}).get("duration_hours", 2.5))
    season = args.season or cfg.get("season", season_from_date(dt.datetime.now().date()))

    # Home / start
    if args.home:
        set_home_by_key(args.home, locs)
    else:
        set_home_default_from_cfg(locs, cfg)

    hh, mm = parse_time_str(depart_str)
    depart_dt = dt.datetime.combine(dt.datetime.now().date(), dt.time(hh, mm))

    results = []
    set_tech_bias_override(tech_bias)
    for loc in locs:
        try:
            r = score_location(loc, depart_dt, duration_h, terrain_bias, max_drive, season, tech_bias=tech_bias)
            results.append(r)
        except Exception as e:
            results.append({
                "key": loc.key, "name": loc.name, "score": 0.0,
                "components": {"weather": 0, "trail": 0, "proximity": 0, "terrain_fit": 0, "secondary": 0},
                "drive_est_min": int(sum(loc.drive_min_typical) / 2),
                "recommend_window": f"{depart_dt:%H:%M}–{(depart_dt + dt.timedelta(hours=duration_h)):%H:%M}",
                "notes": [f"Error fetching/scoring: {e}"]
            })

    print(f"Home: {HOME_KEY} | Depart: {depart_dt:%H:%M} | Max drive: {max_drive} min | Elevation bias: {terrain_bias:+.2f} | Tech bias: {tech_bias:+.2f} | Duration: {duration_h} h | Season: {season}")
    render_output(results, top_n=5)
    print("Alternates:")
    for r in sorted(results, key=lambda x: x["score"], reverse=True)[5:7]:
        print(f" - {r['name']} — {r['score']}")

# --- Module-level load for Streamlit ---
_cfg_module = load_config("config.yaml")
LOCATIONS: List[Location] = load_locations_from_config(_cfg_module) or []
set_home_default_from_cfg(LOCATIONS, _cfg_module)

if __name__ == "__main__":
    main()
