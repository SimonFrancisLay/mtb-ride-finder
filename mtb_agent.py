#!/usr/bin/env python3
"""
mtb_agent.py — HOTFIX
Ensures LOCATIONS and DEFAULT_WEIGHTS are defined at import time.
Includes the moisture-bucket trail model and config.yaml loader.
"""

import argparse, math, requests, datetime as dt, statistics as stats, yaml, os, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

LONDON_TZ = "Europe/London"
TECH_BIAS_OVERRIDE = None
PROX_OVERRIDE = True
WEIGHT_OVERRIDE: Optional[Dict[str, float]] = None  # weather, trail, proximity, terrain_fit, secondary

# --- Caches (TTL 1h) ---
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
    with open(path, "r", encoding="utf-8") as f:
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
    for d in (cfg.get("locations") or []):
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

# ---------- Weather (hourly cached) ----------
def _weather_cache_key(lat: float, lon: float, timezone: str, past_days: int) -> Tuple[float, float, str, int]:
    rl = round(lat, 2); rlon = round(lon, 2)
    hour_key = dt.datetime.now().strftime("%Y%m%d%H") + timezone
    return (rl, rlon, hour_key, int(past_days))

def fetch_open_meteo(lat: float, lon: float, timezone: str, past_days: int = 30, forecast_days: int = 2) -> dict:
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
        "past_days": past_days, "timezone": timezone, "forecast_days": forecast_days
    }
    r = requests.get(base, params=params, timeout=20); r.raise_for_status()
    data = r.json()
    _WEATHER_CACHE[key] = (now, data)
    return data

# ---------- Moisture-bucket model ----------
DRY_PARAMS = {
    "summer": {
        "trail_centre": {"k_dry": 0.45, "beta": 0.045},
        "rocky":        {"k_dry": 0.40, "beta": 0.050},
        "mixed":        {"k_dry": 0.32, "beta": 0.060},
        "mixed_long_mud":{"k_dry": 0.25, "beta": 0.075},
        "moor_slow":    {"k_dry": 0.18, "beta": 0.085},
    },
    "spring": {
        "trail_centre": {"k_dry": 0.35, "beta": 0.050},
        "rocky":        {"k_dry": 0.32, "beta": 0.055},
        "mixed":        {"k_dry": 0.25, "beta": 0.070},
        "mixed_long_mud":{"k_dry": 0.18, "beta": 0.085},
        "moor_slow":    {"k_dry": 0.12, "beta": 0.100},
    },
    "autumn": {
        "trail_centre": {"k_dry": 0.30, "beta": 0.055},
        "rocky":        {"k_dry": 0.26, "beta": 0.060},
        "mixed":        {"k_dry": 0.20, "beta": 0.080},
        "mixed_long_mud":{"k_dry": 0.14, "beta": 0.100},
        "moor_slow":    {"k_dry": 0.10, "beta": 0.120},
    },
    "winter": {
        "trail_centre": {"k_dry": 0.22, "beta": 0.060},
        "rocky":        {"k_dry": 0.20, "beta": 0.070},
        "mixed":        {"k_dry": 0.14, "beta": 0.095},
        "mixed_long_mud":{"k_dry": 0.10, "beta": 0.120},
        "moor_slow":    {"k_dry": 0.08, "beta": 0.140},
    },
}

def _params_for(drainage: str, season: str) -> Tuple[float, float]:
    s = DRY_PARAMS.get(season, DRY_PARAMS["autumn"])
    p = s.get(drainage, s["mixed"])
    return p["k_dry"], p["beta"]

def _effective_recent_mm(hourly: dict, depart_dt: dt.datetime, lookback_h: int = 24) -> float:
    times = [dt.datetime.fromisoformat(t) for t in hourly.get("time", [])]
    vals = hourly.get("precipitation", [])
    if not times or not vals:
        return 0.0
    end = depart_dt; start = depart_dt - dt.timedelta(hours=lookback_h); mm = 0.0
    for t, p in zip(times, vals):
        b = t; e = t + dt.timedelta(hours=1)
        os = max(start, b); oe = min(end, e)
        ov = (oe - os).total_seconds() / 3600.0
        if ov <= 0: continue
        frac = min(1.0, max(0.0, ov))
        centre = os + (oe - os) / 2
        hrs = (end - centre).total_seconds() / 3600.0
        w = 1.6 if hrs <= 6 else (1.0 if hrs <= 12 else 0.7)
        mm += (p or 0.0) * frac * w
    return mm

def _mm_to_moisture(mm: float, beta: float) -> float:
    return 1.0 - math.exp(-beta * max(0.0, float(mm)))

def _step_bucket(prev_M: float, rain_mm: float, k_dry: float, beta: float) -> float:
    M = prev_M * (1.0 - k_dry)
    M = M + (1.0 - M) * _mm_to_moisture(rain_mm, beta)
    return clamp(M, 0.0, 1.0)

def _run_bucket_series(daily_mm: list, end_idx: int, k_dry: float, beta: float, warmup_days: int = 28) -> float:
    if not daily_mm or end_idx < 0:
        return 0.5
    start = max(0, end_idx - warmup_days + 1); M = 0.5
    for i in range(start, end_idx + 1):
        M = _step_bucket(M, daily_mm[i], k_dry, beta)
    return M

def moisture_trail_score(drainage: str, season: str, daily_mm: list, end_idx: int, recent_eff_mm: float = 0.0) -> float:
    k_dry, beta = _params_for(drainage, season)
    M = _run_bucket_series(daily_mm, end_idx, k_dry, beta, 28)
    if recent_eff_mm > 0:
        extra = _mm_to_moisture(recent_eff_mm * {"summer":0.6,"spring":0.8,"autumn":1.0,"winter":1.1}[season], beta)
        M = clamp(M + (1.0 - M) * extra, 0.0, 1.0)
    score = 100.0 * (1.0 - M)
    if recent_eff_mm >= 12.0: score = min(score, 35.0)
    elif recent_eff_mm >= 8.0: score = min(score, 45.0)
    elif recent_eff_mm >= 5.0: score = min(score, 60.0)
    elif recent_eff_mm >= 2.0: score = min(score, 75.0)
    return clamp(score, 0.0, 100.0)

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
    HOME_KEY = key; HOME_COORDS = None
    if key:
        for l in locs:
            if l.key == key: HOME_COORDS = (l.lat, l.lon); return
    for l in locs:
        if l.key == "urmston": HOME_COORDS = (l.lat, l.lon); HOME_KEY = "urmston"; return
    if locs: HOME_COORDS = (locs[0].lat, locs[0].lon); HOME_KEY = locs[0].key

def drive_minutes_from_home(home_key: str, home_lat: float, home_lon: float, loc: Location, depart_hh: int) -> int:
    now = time.time(); k = (home_key or "custom", loc.key, int(depart_hh))
    cached = _DRIVE_CACHE.get(k); 
    if cached and (now - cached[0] < 3600): return cached[1]
    dist_km = _haversine_km(home_lat, home_lon, loc.lat, loc.lon)
    mins = int((dist_km / 65.0) * 60.0)
    mins = int(mins * rush_hour_multiplier(depart_hh))
    mins = max(8, min(600, mins))
    _DRIVE_CACHE[k] = (now, mins); return mins

def score_proximity(est_drive_min: int, max_drive: int) -> float:
    if est_drive_min > max_drive: return 0.0
    return map_range(est_drive_min, 0, max_drive, 100, 0)

# ----- Weight override management -----
def set_weight_override(w: Optional[Dict[str, float]]):
    global WEIGHT_OVERRIDE; WEIGHT_OVERRIDE = w.copy() if w else None

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
    if b > 0:
        score += dist_w * (1.0 if has_hills else 0.0) * abs(b) - (dist_w * 0.32) * (0.0 if has_hills else 1.0) * abs(b)
    elif b < 0:
        score += dist_w * (1.0 if has_distance else 0.0) * abs(b) - (dist_w * 0.32) * (0.0 if has_distance else 1.0) * abs(b)
    tb = clamp(tech_bias, -1.0, 1.0)
    tech_w = dist_w * 1.30
    if tb > 0:
        score += tech_w * (1.0 if has_gnar else 0.0) * abs(tb) - (tech_w * 0.32) * (0.0 if has_gnar else 1.0) * abs(tb)
    elif tb < 0:
        score += tech_w * (1.0 if has_chilled else 0.0) * abs(tb) - (tech_w * 0.32) * (0.0 if has_chilled else 1.0) * abs(tb)
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
    cloudiness = sum(clouds)/len(clouds)
    cloud_pen = map_range(cloudiness, 0, 100, 0, 15)
    tmin, tmax = min(temps), max(temps)
    cold_pen = map_range(5 - tmin, 0, 10, 0, 25) if tmin < 5 else 0
    heat_pen = map_range(tmax - 24, 0, 12, 0, 25) if tmax > 24 else 0
    secondary = 100 - min(40, (cloud_pen + cold_pen + heat_pen))
    return clamp(secondary, 0, 100)

def assemble_reason(loc: Location, total: float, tr_s: float, prox_s: float, terr_s: float, sec_s: float,
                    depart: dt.datetime, duration_h: float, drive_est: int,
                    wind_mean: float, gust_mean: float, rain_prob: float):
    notes = []
    notes.append(f"Trail score {int(tr_s)}/100 (rain adds moisture → lower scores; recovery is gradual).")
    notes.append(f"Wind: {int(wind_mean)} avg (gust {int(gust_mean)}).")
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

# ---------- Main weather scoring ----------
def score_weather(hourly: dict, depart: dt.datetime, duration_h: float) -> float:
    times = [dt.datetime.fromisoformat(t) for t in hourly["time"]]
    end = depart + dt.timedelta(hours=duration_h)
    idx = [i for i, t in enumerate(times) if depart <= t < end]
    if not idx: return 50.0
    def sel(k): return [hourly[k][i] for i in idx]
    avg_prob = sum(sel("precipitation_probability"))/len(idx)
    avg_mm = sum(sel("precipitation"))/len(idx)
    avg_w = sum(sel("wind_speed_10m"))/len(idx)
    avg_g = sum(sel("wind_gusts_10m"))/len(idx)
    avg_cloud = sum(sel("cloudcover"))/len(idx)
    avg_temp = sum(sel("temperature_2m"))/len(idx)
    rain_penalty = clamp(avg_prob * (1 + avg_mm), 0, 100)
    rain_score = 100 - map_range(rain_penalty, 0, 100, 0, 60)
    wind_index = avg_w * 0.6 + avg_g * 0.4
    wind_score = 100 - map_range(wind_index, 0, 50, 0, 50)
    sun_score = map_range(100 - avg_cloud, 0, 100, 40, 100)
    cold_pen = map_range(10 - avg_temp, 0, 15, 0, 25) if avg_temp < 10 else 0
    heat_pen = map_range(avg_temp - 22, 0, 12, 0, 25) if avg_temp > 22 else 0
    temp_score = 100 - min(40, (cold_pen + heat_pen))
    return clamp(0.5 * rain_score + 0.3 * wind_score + 0.1 * sun_score + 0.1 * temp_score, 0, 100)

# Helpers
def _find_day_index(daily_times: list, target_date: dt.date) -> int:
    for i, iso in enumerate(daily_times or []):
        try:
            d = dt.date.fromisoformat(iso)
            if d == target_date:
                return i
        except Exception:
            pass
    return -1

def trail_score_for(loc: Location, season: str, data: dict, target_date: dt.date, depart_dt: dt.datetime, mode: str = "time_aware") -> float:
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})
    precip = list(daily.get("precipitation_sum", []))
    times = daily.get("time", [])
    if not precip or not times: return 50.0
    idx = _find_day_index(times, target_date)
    if idx < 0: idx = len(precip) - 1
    recent_eff = _effective_recent_mm(hourly, depart_dt, 24) if mode == "time_aware" else 0.0
    season_eff = season_from_date(target_date) if season == "auto" else season
    return moisture_trail_score(loc.drainage, season_eff, precip, idx, recent_eff)

def trail_condition_for_date(loc: Location, season: str, target_dt: dt.datetime, mode: str) -> float:
    data = fetch_open_meteo(loc.lat, loc.lon, LONDON_TZ, past_days=30, forecast_days=2)
    return trail_score_for(loc, season, data, target_dt.date(), target_dt, mode=mode)

# ----- Score a single location -----
def score_location(loc: Location, when_dt: dt.datetime, duration_h: float, terrain_bias: float, max_drive: int,
                   season: str, tech_bias: float = None, home_key: Optional[str] = None,
                   home_coords: Optional[Tuple[float, float]] = None, weights: Optional[Dict[str, float]] = None,
                   trail_mode: str = "time_aware"):
    data = fetch_open_meteo(loc.lat, loc.lon, LONDON_TZ, past_days=30, forecast_days=2)
    hourly = data["hourly"]
    wx_s = score_weather(hourly, when_dt, duration_h)
    tr_s = trail_score_for(loc, season, data, when_dt.date(), when_dt, mode=trail_mode)
    hk = None; hcoords = None
    if home_key is not None or home_coords is not None:
        hk = home_key or "custom"; hcoords = home_coords
    elif HOME_COORDS is not None:
        hk, hcoords = (HOME_KEY or "custom"), HOME_COORDS
    if hcoords is not None:
        drive_est = drive_minutes_from_home(hk, hcoords[0], hcoords[1], loc, when_dt.hour)
    else:
        drive_mid = sum(loc.drive_min_typical) / 2
        drive_est = int(drive_mid * rush_hour_multiplier(when_dt.hour))
    prox_s = score_proximity(drive_est, max_drive)
    tb = TECH_BIAS_OVERRIDE if TECH_BIAS_OVERRIDE is not None else (tech_bias or 0.0)
    terr_s = score_terrain_fit(loc.terrain, terrain_bias, tb, duration_h, loc.duration_range)
    sec_s = score_secondary(hourly, when_dt, duration_h)
    use_w = weights or WEIGHT_OVERRIDE
    if not use_w:
        use_w = {"weather": 0.25, "trail": 0.35, "proximity": 0.10, "terrain_fit": 0.25, "secondary": 0.05}
    if PROX_OVERRIDE and terr_s >= 70 and tr_s >= 70:
        prox_s = max(prox_s, 50)
    total = (use_w["weather"] * wx_s + use_w["trail"] * tr_s + use_w["proximity"] * prox_s +
             use_w["terrain_fit"] * terr_s + use_w["secondary"] * sec_s)
    times_h = [dt.datetime.fromisoformat(t) for t in hourly["time"]]
    end = when_dt + dt.timedelta(hours=duration_h)
    idx = [i for i, t in enumerate(times_h) if when_dt <= t < end]
    if idx:
        wind_mean = sum(hourly["wind_speed_10m"][i] for i in idx) / len(idx)
        gust_mean = sum(hourly["wind_gusts_10m"][i] for i in idx) / len(idx)
        rain_prob = sum(hourly["precipitation_probability"][i] for i in idx) / len(idx)
    else:
        wind_mean = gust_mean = rain_prob = 0.0
    reason = assemble_reason(loc, total, tr_s, prox_s, terr_s, sec_s, when_dt, duration_h, drive_est, wind_mean, gust_mean, rain_prob)
    return {
        "key": loc.key, "name": loc.name, "score": round(total, 1),
        "components": {"weather": round(wx_s, 1), "trail": round(tr_s, 1), "proximity": round(prox_s, 1),
                       "terrain_fit": round(terr_s, 1), "secondary": round(sec_s, 1)},
        "drive_est_min": int(drive_est),
        "recommend_window": f"{when_dt:%H:%M}–{(when_dt + dt.timedelta(hours=duration_h)):%H:%M}",
        "notes": reason
    }

# ----- Trail condition history -----
def trail_condition_series(loc: Location, season: str, days: int = 10, window: int = 5) -> List[float]:
    hour_key = dt.datetime.now().strftime("%Y%m%d%H")
    ck = (loc.key + hour_key, days, window, season)
    now = time.time()
    cached = _TRAIL_SERIES_CACHE.get(ck)
    if cached and (now - cached[0] < 3600):
        return cached[1]
    data = fetch_open_meteo(loc.lat, loc.lon, LONDON_TZ, past_days=max(30, days + 28), forecast_days=0)
    daily = data.get("daily", {})
    precip = daily.get("precipitation_sum", [])
    if not precip or len(precip) < days + 5:
        res = [50.0] * days; _TRAIL_SERIES_CACHE[ck] = (now, res); return res
    upto = len(precip) - 1
    start = max(0, upto - days + 1)
    series = []
    k_dry, beta = _params_for(loc.drainage, season)
    M = 0.5
    warm_start = max(0, start - 28)
    for i in range(warm_start, start):
        M = _step_bucket(M, precip[i], k_dry, beta)
    for i in range(start, upto + 1):
        M = _step_bucket(M, precip[i], k_dry, beta)
        score = 100.0 * (1.0 - M)
        series.append(round(clamp(score, 0.0, 100.0), 1))
    series = series[-days:]
    _TRAIL_SERIES_CACHE[ck] = (now, series)
    return series

# ----- Outlook wrapper -----
def trail_condition_for_date_outlook(loc: Location, season: str, target_dt: dt.datetime, mode: str) -> float:
    return trail_condition_for_date(loc, season, target_dt, mode=mode)

def set_home_default_from_cfg(locs: List[Location], cfg: dict):
    for l in locs:
        if l.key == "urmston":
            set_home_by_key("urmston", locs); return
    if locs: set_home_by_key(locs[0].key, locs)

# --- Module-level: load config + expose LOCATIONS + DEFAULT_WEIGHTS ---
try:
    _cfg = load_config("config.yaml")
except Exception:
    _cfg = {}

LOCATIONS: List[Location] = load_locations_from_config(_cfg) if _cfg else []
DEFAULT_WEIGHTS: Dict[str, float] = get_weights_from_config(_cfg) if _cfg else {
    "weather": 0.25, "trail": 0.35, "proximity": 0.10, "terrain_fit": 0.25, "secondary": 0.05
}
set_home_default_from_cfg(LOCATIONS, _cfg or {})

# ----- CLI for quick local test -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=str, default=None)
    parser.add_argument("--depart", type=str, default="08:00")
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--max-drive", type=int, default=90)
    parser.add_argument("--days-ahead", type=int, default=0)
    parser.add_argument("--terrain-bias", type=float, default=0.0)
    parser.add_argument("--tech-bias", type=float, default=0.0)
    args = parser.parse_args()

    if args.home:
        set_home_by_key(args.home, LOCATIONS)

    hh, mm = parse_time_str(args.depart)
    base = dt.datetime.combine(dt.date.today(), dt.time(hh, mm))
    when = base + dt.timedelta(days=args.days_ahead)
    mode = "time_aware" if args.days_ahead <= 1 else "daily"

    for loc in LOCATIONS:
        r = score_location(loc, when, args.duration, args.terrain_bias, args.max_drive, season_from_date(when.date()),
                           tech_bias=args.tech_bias, trail_mode=mode)
        print(when.date(), loc.name, r["score"], r["components"], r["drive_est_min"])

if __name__ == "__main__":
    main()
