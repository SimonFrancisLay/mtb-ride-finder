# mtb_agent.py — consolidated, with legacy History + time-aware Outlook/Top Picks
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math, os, datetime as dt, hashlib

try:
    import yaml
except Exception:
    yaml = None

# ---------------- Config loading ----------------
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def _load_yaml(path: str) -> dict:
    if not os.path.exists(path) or yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def reload_config() -> None:
    """Force reload of config.yaml and refresh global LOCATIONS + DEFAULT_WEIGHTS."""
    global LOCATIONS, DEFAULT_WEIGHTS
    data = _load_yaml(CFG_PATH)
    DEFAULT_WEIGHTS = data.get("weights", DEFAULT_WEIGHTS)
    new_locs = []
    for d in data.get("locations", []):
        try:
            new_locs.append(_location_from_dict(d))
        except Exception:
            pass
    LOCATIONS = new_locs

@dataclass
class Location:
    key: str
    region: str
    name: str
    lat: float
    lon: float
    drive_min_typical: Tuple[int, int]
    terrain: str
    drainage: str
    mud_sensitivity: str
    duration_range: Tuple[float, float]
    notes: str = ""

def _location_from_dict(d: dict) -> Location:
    return Location(
        key=d["key"],
        region=d.get("region",""),
        name=d["name"],
        lat=float(d["lat"]),
        lon=float(d["lon"]),
        drive_min_typical=tuple(d.get("drive_min_typical",[0,0])),
        terrain=d.get("terrain",""),
        drainage=d.get("drainage","mixed"),
        mud_sensitivity=d.get("mud_sensitivity","med"),
        duration_range=tuple(d.get("duration_range",[2.0,4.0])),
        notes=d.get("notes","")
    )

_cfg = _load_yaml(CFG_PATH)

DEFAULT_WEIGHTS: Dict[str, float] = _cfg.get("weights", {
    "weather": 0.25, "trail": 0.35, "proximity": 0.10, "terrain_fit": 0.25, "secondary": 0.05
})

LOCATIONS: List[Location] = []
for d in _cfg.get("locations", []):
    try:
        LOCATIONS.append(_location_from_dict(d))
    except Exception:
        pass

# ---------------- Runtime overrides (set from the app) ----------------
_HOME_KEY: Optional[str] = None
_TECH_BIAS: Optional[float] = None
_PROX_OVERRIDE: bool = True
_WEIGHTS_OVERRIDE: Optional[Dict[str, float]] = None

def set_home_by_key(key: str, locs: List[Location]) -> None:
    global _HOME_KEY
    _HOME_KEY = key if any(l.key == key for l in locs) else None

def set_tech_bias_override(val: Optional[float]) -> None:
    global _TECH_BIAS
    _TECH_BIAS = val

def set_prox_override(flag: bool) -> None:
    global _PROX_OVERRIDE
    _PROX_OVERRIDE = bool(flag)

def set_weight_override(weights: Optional[Dict[str, float]]) -> None:
    global _WEIGHTS_OVERRIDE
    _WEIGHTS_OVERRIDE = weights

# ---------------- Helpers ----------------
def season_from_date(d: dt.date) -> str:
    m = d.month
    if m in (12,1,2): return "winter"
    if m in (3,4,5): return "spring"
    if m in (6,7,8): return "summer"
    return "autumn"

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _home_latlon() -> Tuple[float,float]:
    if not LOCATIONS:
        return (53.48, -2.3)
    if _HOME_KEY:
        for l in LOCATIONS:
            if l.key == _HOME_KEY:
                return (l.lat, l.lon)
    for l in LOCATIONS:
        if l.key == "urmston":
            return (l.lat, l.lon)
    l0 = LOCATIONS[0]
    return (l0.lat, l0.lon)

# ---------------- Time-aware dryness proxy (fast, deterministic) ----------------
def _deterministic_score(lat: float, lon: float, when: dt.datetime, season: str) -> float:
    base = {"winter": 55, "spring": 65, "summer": 80, "autumn": 60}.get(season, 60)
    jitter = ((lat*7.3 + lon*11.1) % 5)                # small per-location variation
    hour = when.hour + when.minute/60.0
    diurnal = 2.5 * math.cos((hour-10)/24 * 2*math.pi) # gentle time-of-day variation
    return max(0.0, min(100.0, base + jitter + diurnal))

def trail_condition_for_date_outlook(loc: Location, season: str, when: dt.datetime, mode: str="time_aware") -> float:
    return round(_deterministic_score(loc.lat, loc.lon, when, season), 1)

# ---------------- Legacy History model (rainfall + decay + seasonal caps) ----------------
def _seeded_rand01(*parts) -> float:
    s = "|".join(str(p) for p in parts).encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    v = int(h[:8], 16)
    return (v % 10_000) / 10_000.0

def _synthetic_rain_mm(loc: Location, day: dt.date, season: str) -> float:
    seasonal = {"winter": 5.5, "spring": 3.0, "summer": 2.2, "autumn": 4.2}.get(season, 3.5)
    noise = (_seeded_rand01(loc.key, day.toordinal()) - 0.5) * 6.0  # ±3mm
    storm = 0.0
    if _seeded_rand01("storm", loc.key, day.toordinal()) > 0.92:
        storm = 8.0 + 20.0 * _seeded_rand01("stormmag", loc.key, day.toordinal())
    return max(0.0, seasonal + noise + storm)

def trail_condition_series_legacy(loc: Location, season: str, days: int = 10, window: int = 5) -> List[float]:
    dry_cap = {"winter": 80, "spring": 90, "summer": 100, "autumn": 85}.get(season, 85)
    drain = (loc.drainage or "mixed").lower()
    mud = (loc.mud_sensitivity or "med").lower()

    rec = 1.0
    if drain in ("trail_centre","rocky"):
        rec *= 1.05
    elif drain in ("mixed",):
        rec *= 1.00
    else:
        rec *= 0.90  # mixed_long_mud / peaty
    if mud == "low":
        rec *= 1.05
    elif mud == "med":
        rec *= 1.00
    else:
        rec *= 0.92

    today = dt.date.today()
    out: List[float] = []
    for i in range(days, 0, -1):
        day = today - dt.timedelta(days=i-1)
        total = 0.0
        for k in range(1, window+1):
            total += _synthetic_rain_mm(loc, day - dt.timedelta(days=k), season)
        wet = 100.0 * (1.0 - math.exp(-total / 40.0))   # diminishing returns
        dry = (100.0 - wet) * rec
        dry = max(0.0, min(dry_cap, dry))
        out.append(round(dry, 1))
    return out

# ---------------- Scoring ----------------
def _terrain_fit_score(loc: Location, pref_elev: float, tech_bias: Optional[float]) -> float:
    tags = {t.strip().lower() for t in loc.terrain.split(",") if t.strip()}
    elev_pref = pref_elev  # -1..+1
    elev_score = 50
    if elev_pref > 0:
        elev_score += 25 if ("hills" in tags or "steep" in tags) else -10
    elif elev_pref < 0:
        elev_score += 25 if ("distance" in tags or "flat" in tags or "gravel" in tags) else -10

    tech_score = 50
    tb = _TECH_BIAS if tech_bias is None else tech_bias
    if tb is not None and tb != 0:
        if tb > 0:  # wants gnar
            tech_score += 25 if ("technical" in tags or "steep" in tags or "high_tech" in tags) else -10
        else:       # wants chilled
            tech_score += 25 if ("gravel" in tags or "trail_centre" in tags or "distance" in tags or "flat" in tags) else -10

    return max(0.0, min(100.0, 0.7*elev_score + 0.3*tech_score))

def _proximity_score_and_drive(loc: Location) -> Tuple[float, int]:
    hlat, hlon = _home_latlon()
    km = _haversine_km(hlat, hlon, loc.lat, loc.lon)
    drive_min = int(km / 70.0 * 60.0)  # ~70 km/h avg speed
    prox = max(0.0, 100.0 - (drive_min/180.0)*100.0)  # 0..180+ min → 100..0
    return (round(prox,1), drive_min)

def score_location(loc: Location, when_dt: dt.datetime, duration_h: float, pref_elev: float,
                   max_drive_min: int, season: str, tech_bias: Optional[float]=None,
                   weights: Optional[Dict[str,float]]=None, trail_mode: str="time_aware") -> Dict:
    ww = (weights or _WEIGHTS_OVERRIDE or DEFAULT_WEIGHTS).copy()

    weather = trail_condition_for_date_outlook(loc, season, when_dt, mode=trail_mode)
    legacy5 = trail_condition_series_legacy(loc, season, days=5, window=5)[-1]
    trail = round(0.6*weather + 0.4*legacy5, 1)

    terrain_fit = _terrain_fit_score(loc, pref_elev, _TECH_BIAS if tech_bias is None else tech_bias)
    proximity, drive_est = _proximity_score_and_drive(loc)
    secondary = 50.0

    if _PROX_OVERRIDE and (terrain_fit > 70 and trail > 65):
        proximity = max(proximity, 60.0)

    total = sum(ww.values()) or 1.0
    wnorm = {k: v/total for k,v in ww.items()}
    score = (wnorm["weather"]*weather + wnorm["trail"]*trail +
             wnorm["proximity"]*proximity + wnorm["terrain_fit"]*terrain_fit +
             wnorm["secondary"]*secondary)
    score = round(score, 1)

    notes = [
        f"Live dryness {weather}/100; legacy recent {legacy5}/100 → trail {trail}/100.",
        f"Terrain fit {terrain_fit}/100, proximity {proximity}/100 (drive ≈ {drive_est} min)."
    ]
    window = f"{when_dt:%H:%M}–{(when_dt + dt.timedelta(hours=duration_h)):%H:%M}"
    return {
        "key": loc.key,
        "name": loc.name,
        "score": score,
        "components": {
            "weather": round(weather,1),
            "trail": round(trail,1),
            "proximity": round(proximity,1),
            "terrain_fit": round(terrain_fit,1),
            "secondary": round(secondary,1),
        },
        "drive_est_min": drive_est,
        "recommend_window": window,
        "notes": notes,
    }
