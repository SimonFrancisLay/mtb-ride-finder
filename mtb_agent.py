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
WEIGHT_OVERRIDE: Optional[Dict[str, float]] = None  # weather, trail, proximity, terrain_fit, secondary

# --- Hourly caches (TTL ~ 3600s) ---
_WEATHER_CACHE: Dict[Tuple[float, float, str, int], Tuple[float, dict]] = {}
_DRIVE_CACHE: Dict[Tuple[str, str, int], Tuple[float, int]] = {}
_TRAIL_SERIES_CACHE: Dict[Tuple[str, int, int, str], Tuple[float, List[float]]] = {}

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

# ---------- Weather / data (hourly cached) ----------
def _weather_cache_key(lat: float, lon: float, timezone: str, past_days: int) -> Tuple[float, float, str, int]:
    rl = round(lat, 2); rlon = round(lon, 2)
    hour_key = dt.datetime.now().strftime("%Y%m%d%H") + timezone
    return (rl, rlon, hour_key, int(past_days))

def fetch_open_meteo(lat: float, lon: float, timezone: str, past_days: int = 7, forecast_days: int = 2) -> dict:
    key = _weather_cache_key(lat, lon, timezone, past_days)
    now = time.time()
    cached = _WEATHER_CACHE.get(key)
    if cached and (now - cached[0] < 3600):
        return cached[1]
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m,cloudcover,temperature_2m",
        "daily": "precipitation_sum,sunshine_duration,temperature_2m_max,temperature_2m_min",
        "past_days": past_days, "timezone": timezone
    }
    if forecast_days and forecast_days > 0:
        params["forecast_days"] = forecast_days
    r = requests.get(base, params=params, timeout=20); r.raise_for_status()
    data = r.json()
    _WEATHER_CACHE[key] = (now, data)
    return data

# ---------- Trail dryness (weighted + hard cap) ----------
def _drain_factor(drainage: str) -> float:
    if drainage in ("trail_centre", "rocky"):
        return 1.0
    if drainage in ("mixed_long_mud", "moor_slow"):
        return 1.5
    return 1.2  # "mixed" fallback

def _season_base_days(season: str) -> float:
    if season == "summer": return 5.0
    if season == "winter": return 10.0
    return 7.5  # spring/autumn

def _consecutive_dry_days(daily_mm: list, end_idx: int, dry_thresh_mm: float = 1.0, lookback: int = 30) -> int:
    cnt = 0
    i = end_idx
    while i >= 0 and cnt < lookback:
        if daily_mm[i] <= dry_thresh_mm:
            cnt += 1
            i -= 1
        else:
            break
    return cnt

def evaluate_trail_dryness(drainage: str, season: str, daily_mm: list, end_idx: int, window: int = 5) -> float:
    if not daily_mm or end_idx < 0:
        return 50.0
    base_weights = [0.40, 0.25, 0.18, 0.11, 0.06, 0.04, 0.02]
    w = base_weights[:max(1, min(window, len(base_weights)))]
    w_sum = sum(w); w = [x / w_sum for x in w]
    idx0 = max(0, end_idx - (len(w) - 1))
    mm_window = daily_mm[idx0:end_idx + 1]
    if len(mm_window) < 1:
        return 50.0
    w = w[-len(mm_window):]
    weighted_mm = sum(m * ww for m, ww in zip(mm_window[::-1], w))  # most recent gets highest weight
    k_map = {"trail_centre": 0.028, "rocky": 0.032, "mixed": 0.040, "mixed_long_mud": 0.050, "moor_slow": 0.060}
    k = k_map.get(drainage, 0.040)
    base_score = clamp(100.0 * math.exp(-k * weighted_mm), 0.0, 100.0)
    base_days = _season_base_days(season)
    factor = _drain_factor(drainage)
    days_needed = base_days * factor
    dry_days = _consecutive_dry_days(daily_mm, end_idx, dry_thresh_mm=1.0, lookback=30)
    cap_max = 60.0 + 40.0 * clamp(dry_days / max(1.0, days_needed), 0.0, 1.0)
    return min(base_score, cap_max)

# ---------- Drive & proximity ----------
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

HOME_KEY: Optional[str] = None
HOME_COORDS: Optional[Tuple[float, float]] = None

def set_home_by_key(key: Optional[str], locs: List[Location]) -> None:
    global HOME_KEY, HOME_COORDS
    HOME_KEY = key
    HOME_COORDS = None
    if key:
        for l in locs:
            if l.key == key:
                HOME_COORDS = (l.lat, l.lon)
                return
    for l in locs:
        if l.key == "urmston":
            HOME_COORDS = (l.lat, l.lon); HOME_KEY = "urmston"; return
    if locs:
        HOME_COORDS = (locs[0].lat, locs[0].lon); HOME_KEY = locs[0].key

def drive_minutes_from_home(home_key: str, home_lat: float, home_lon: float, loc: Location, depart_hh: int) -> int:
    now = time.time()
    k = (home_key or "custom", loc.key, int(depart_hh))
    cached = _DRIVE_CACHE.get(k)
    if cached and (now - cached[0] < 3600):
        return cached[1]
    dist_km = _haversine_km(home_lat, home_lon, loc.lat, loc.lon)
    mins = int((dist_km / 65.0) * 60.0)
    mins = int(mins * rush_hour_multiplier(depart_hh))
    mins = max(8, min(600, mins))
    _DRIVE_CACHE[k] = (now, mins)
    return mins

def score_proximity(est_drive_min: int, max_drive: int) -> float:
    if est_drive_min > max_drive: return 0.0
    return map_range(est_drive_min, 0, max_drive, 100, 0)

# ----- Weight override management -----
def set_weight_override(w: Optional[Dict[str, float]]):
    global WEIGHT_OVERRIDE
    WEIGHT_OVERRIDE = w.copy() if w else None

def get_weights_from_config(cfg: dict) -> Dict[str, float]:
    w = cfg.get("weights", {})
    return {
        "weather": float(w.get("weather", 0.25)),
        "trail": float(w.get("trail", 0.35)),
        "proximity": float(w.get("proximity", 0.10)),
        "terrain_fit": float(w.get("terrain_fit", 0.25)),
        "secondary": float(w.get("secondary", 0.05)),
    }

# ---------- Terrain fit & secondary ----------
def score_terrain_fit(terrain_tags: str, pref_bias: float, tech_bias: float, duration_h: float, loc_range: Tuple[float, float]) -> float:
    tags = set([t.strip() for t in terrain_tags.split(",")])
    has_hills = any(t in tags for t in ["hills", "steep", "mod_elev"])
    has_distance = any(t in tags for t in ["distance", "flat", "gravel"])
    has_gnar = any(t in tags for t in ["technical", "high_tech", "steep"])
    has_chilled = any(t in tags for t in ["gravel", "flat", "trail_centre", "distance"])
    score = 30.0
    b = clamp(pref_bias, -1.0, 1.0)
    dist_w = 25.0
    if b > 0: score += dist_w * (1.0 if has_hills else 0.0) * abs(b) - (dist_w * 0.32) * (0.0 if has_hills else 1.0) * abs(b)
    elif b < 0: score += dist_w * (1.0 if has_distance else 0.0) * abs(b) - (dist_w * 0.32) * (0.0 if has_distance else 1.0) * abs(b)
    tb = clamp(tech_bias, -1.0, 1.0)
    tech_w = dist_w * 1.30
    if tb > 0: score += tech_w * (1.0 if has_gnar else 0.0) * abs(tb) - (tech_w * 0.32) * (0.0 if has_gnar else 1.0) * abs(tb)
    elif tb < 0: score += tech_w * (1.0 if has_chilled else 0.0) * abs(tb) - (tech_w * 0.32) * (0.0 if has_chilled else 1.0) * abs(tb)
    lo, hi = loc_range
    if lo - 0.25 <= duration_h <= hi + 0.25: score += 20
    elif duration_h < lo: score -= map_range(lo - duration_h, 0, 2, 0, 20)
    else: score -= map_range(duration_h - hi, 0, 2, 0, 20)
    return clamp(score, 0, 100)

def score_secondary(hourly: dict, depart: dt.datetime, duration_h: float) -> float:
    times = [dt.datetime.fromisoformat(t) for t in hourly["time"]]
    end = depart + dt.timedelta(hours=duration_h)
    idx = [i for i, t in enumerate(times) if depart <= t < end]
    if not idx: return 50.0
    def sel(key): return [hourly[key][i] for i in idx]
    temps = sel("temperature_2m"); clouds = sel("cloudcover")
    cloudiness = stats.mean(clouds)
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
    notes.append(f"Dryness driven by recent rainfall → trail {int(tr_s)}/100.")
    notes.append(f"Wind: {int(wind_mean)} avg (gust {int(gust_mean)}); Weather score {int(wx_s)}.")
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

# ---------- Main scoring ----------
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
    weights: Optional[Dict[str, float]] = None,
):
    data = fetch_open_meteo(loc.lat, loc.lon, LONDON_TZ, past_days=15, forecast_days=2)
    hourly = data["hourly"]; daily = data["daily"]
    wx_s = score_weather(hourly, depart_dt, duration_h)

    precip_list = daily.get("precipitation_sum", [])
    # Use yesterday as the endpoint for trail dryness to avoid overreacting to forecast uncertainty
    tr_s = evaluate_trail_dryness(loc.drainage, season, precip_list, end_idx=len(precip_list)-2, window=5)

    hk = None; hcoords = None
    if home_key is not None or home_coords is not None:
        hk = home_key or "custom"; hcoords = home_coords
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

    use_w = weights or WEIGHT_OVERRIDE
    if not use_w:
        use_w = {"weather": 0.25, "trail": 0.35, "proximity": 0.10, "terrain_fit": 0.25, "secondary": 0.05}

    if PROX_OVERRIDE and terr_s >= 70 and tr_s >= 70:
        prox_s = max(prox_s, 50)

    total = (use_w["weather"] * wx_s + use_w["trail"] * tr_s + use_w["proximity"] * prox_s +
             use_w["terrain_fit"] * terr_s + use_w["secondary"] * sec_s)

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
                             drive_est, 0.0, wind_mean, gust_mean, rain_prob)

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

# ----- Trail condition history (10 days) -----
def trail_condition_series(loc: Location, season: str, days: int = 10, window: int = 5) -> List[float]:
    hour_key = dt.datetime.now().strftime("%Y%m%d%H")
    ck = (loc.key + hour_key, days, window, season)
    now = time.time()
    cached = _TRAIL_SERIES_CACHE.get(ck)
    if cached and (now - cached[0] < 3600):
        return cached[1]

    data = fetch_open_meteo(loc.lat, loc.lon, LONDON_TZ, past_days=max(30, days + window + 10), forecast_days=0)
    daily = data.get("daily", {})
    precip = daily.get("precipitation_sum", [])
    if not precip or len(precip) < days + window:
        res = [50.0] * days
        _TRAIL_SERIES_CACHE[ck] = (now, res)
        return res

    upto = len(precip) - 1  # exclude "today"
    start = max(window - 1, upto - days)
    series = []
    for end_idx in range(start, upto):
        score = evaluate_trail_dryness(loc.drainage, season, precip, end_idx=end_idx, window=window)
        series.append(round(score, 1))
    series = series[-days:]
    _TRAIL_SERIES_CACHE[ck] = (now, series)
    return series

# ----- Outlook today & tomorrow (uses forecast) -----
def _find_day_index(daily_times: list, target_date: dt.date) -> int:
    for i, iso in enumerate(daily_times or []):
        try:
            d = dt.date.fromisoformat(iso)
            if d == target_date:
                return i
        except Exception:
            pass
    return -1

def trail_condition_outlook(loc: Location, season: str, window: int = 5) -> Dict[str, float]:
    data = fetch_open_meteo(loc.lat, loc.lon, LONDON_TZ, past_days=15, forecast_days=2)
    daily = data.get("daily", {})
    precip = daily.get("precipitation_sum", [])
    times = daily.get("time", [])
    if not precip or not times:
        return {"today": 50.0, "tomorrow": 50.0}
    today = dt.date.today()
    idx_today = _find_day_index(times, today)
    idx_tom = _find_day_index(times, today + dt.timedelta(days=1))
    out = {}
    out["today"] = round(evaluate_trail_dryness(loc.drainage, season, precip, end_idx=idx_today if idx_today>=0 else len(precip)-2, window=window), 1)
    out["tomorrow"] = round(evaluate_trail_dryness(loc.drainage, season, precip, end_idx=idx_tom if idx_tom>=0 else len(precip)-1, window=window), 1)
    return out

def set_home_default_from_cfg(locs: List[Location], cfg: dict):
    home_label = cfg.get("start_location", "Urmston")
    for l in locs:
        if l.key == "urmston":
            set_home_by_key("urmston", locs); return
    for l in locs:
        if home_label.lower() in l.name.lower():
            set_home_by_key(l.key, locs); return
    if locs:
        set_home_by_key(locs[0].key, locs)

# --- Module-level load for Streamlit ---
_cfg_module = load_config("config.yaml")
LOCATIONS: List[Location] = load_locations_from_config(_cfg_module) or []
DEFAULT_WEIGHTS = get_weights_from_config(_cfg_module)
set_home_default_from_cfg(LOCATIONS, _cfg_module)

# ----- CLI for local testing -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=str, default=None)
    parser.add_argument("--depart", type=str, default="08:00")
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--max-drive", type=int, default=90)
    parser.add_argument("--terrain-bias", type=float, default=0.0)
    parser.add_argument("--tech-bias", type=float, default=0.0)
    args = parser.parse_args()

    if args.home:
        set_home_by_key(args.home, LOCATIONS)

    hh, mm = parse_time_str(args.depart)
    depart_dt = dt.datetime.combine(dt.date.today(), dt.time(hh, mm))

    for loc in LOCATIONS:
        r = score_location(loc, depart_dt, args.duration, args.terrain_bias, args.max_drive, season_from_date(dt.date.today()), tech_bias=args.tech_bias)
        print(loc.name, r["score"], r["components"], r["drive_est_min"])

if __name__ == "__main__":
    main()
