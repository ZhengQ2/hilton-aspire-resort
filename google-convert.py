#!/usr/bin/env python3
"""
Geocode Hilton hotels using Google **Places API** (Text Search + Place Details),
with fallback to the classic Geocoding API only if Places yields nothing acceptable.

Inputs
- CSV with columns: hotel_name, group_label; optional: hotel_url

Outputs
- CSV with precise coordinates (favoring rooftop-quality Place Details),
  cached results, and status info per row.

Env
- GOOGLE_MAPS_API_KEY  (same as before)

Notes
- We now **keep** discriminators like "Hotel" / "Resort" in the query.
- We accept results only if they look like lodging (or a clear property premise).
- Results are cached by the query set to reduce API calls.
"""

import os
import json
import time
import re
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urlparse, unquote

import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# Config (no CLI args)
# -----------------------
INPUT_PATH  = "hilton_resort_credit_hotels_by_brand.csv"  # must include hotel_name, group_label; optional hotel_url
OUTPUT_PATH = "hotels_geocoded_google.csv"
CACHE_PATH  = "geocode_cache_google.json"

REGION_BIAS = ""       # e.g., "US", "CA", "GB" or "" for none (used by Geocoding fallback)
BASE_DELAY  = 0.0       # seconds between requests (jittered)
ONLY_MISSING = False    # if True, skip rows already having lat/lon in OUTPUT_PATH
OFFSET = 0              # start after this many rows (post-filter)
MAX_ROWS = 0            # process at most this many (0 = all)

# --- Google endpoints ---
PLACES_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS_URL     = "https://maps.googleapis.com/maps/api/place/details/json"
GEOCODE_URL            = "https://maps.googleapis.com/maps/api/geocode/json"  # fallback only
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

CACHE_VERSION = 3  # bumped for Places pipeline

class GeocodeError(Exception):
    pass

# -----------------------
# Query building
# -----------------------

def clean_query(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    t = re.sub(r"[®™]", "", t)
    # Remove a very specific noise phrase
    t = re.sub(r"\bat\s+Resorts\s+World\b", "", t, flags=re.I)
    return t.strip(" ,-")

# Keep discriminators like hotel/resort; only drop true glue words
STOPWORDS = {"by","the","and","at","&"}


def _slug_words_from_url(url: str) -> List[str]:
    if not url:
        return []
    try:
        p = urlparse(url)
        segs = [s for s in p.path.split("/") if s]
        if not segs:
            return []
        slug = unquote(segs[-1])
        slug = re.sub(r"[^A-Za-z0-9\- ]+", " ", slug)
        words = re.split(r"[\s\-]+", slug)
        return [w for w in words if w]
    except Exception:
        return []


def _title_like_from_words(words: List[str]) -> str:
    if not words:
        return ""
    # Don't over-strip short slugs; keep most words to retain specificity
    keep = [w for w in words if (len(words) <= 3 or w.lower() not in STOPWORDS)]
    s = " ".join(keep) if keep else " ".join(words)
    s = re.sub(r"\bHilton\b", "Hilton", s, flags=re.I)
    return s.strip()


def build_queries(hotel_name: str, brand: str, hotel_url: str = "") -> List[str]:
    """Build strongest → weakest search strings for Places Text Search.
    We keep discriminators and try brand spellings.
    """
    queries: List[str] = []

    def add(q: str):
        q = clean_query(q)
        if q and q not in queries:
            queries.append(q)

    slug_title = _title_like_from_words(_slug_words_from_url(hotel_url))
    brand_tokens = [brand, "Hilton"] if brand and brand.lower() != "hilton" else ["Hilton"]
    name_candidates = [s for s in [slug_title, hotel_name] if s]

    # Strong variants: include Hotel/Resort suffixes explicitly to skew toward lodging
    for name in name_candidates:
        for b in brand_tokens:
            add(f"{name} {b} Hotel")
            add(f"{name} {b} Resort")
            add(f"{name} {b}")

    # Fallbacks, progressively weaker
    if hotel_name:
        add(f"{hotel_name} Hotel")
        add(f"{hotel_name} Resort")
        add(hotel_name)

    return queries


# -----------------------
# Google API helpers (Places-first)
# -----------------------

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.8, min=1, max=6),
    retry=retry_if_exception_type((requests.RequestException, GeocodeError)),
)
def places_text_search(query: str, api_key: str) -> Dict[str, Any]:
    if not api_key:
        raise GeocodeError("Missing GOOGLE_MAPS_API_KEY")
    params = {"query": query, "key": api_key}
    r = requests.get(PLACES_TEXT_SEARCH_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    status = data.get("status", "")
    if status == "OK":
        return data
    if status in ("ZERO_RESULTS",):
        return {"results": []}
    msg = data.get("error_message") or status or "UNKNOWN"
    raise GeocodeError(f"Places Text Search error: {msg}")


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.8, min=1, max=6),
    retry=retry_if_exception_type((requests.RequestException, GeocodeError)),
)
def places_details(place_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    params = {
        "place_id": place_id,
        "key": api_key,
        "fields": "place_id,name,geometry,formatted_address,types,business_status,website"
    }
    r = requests.get(PLACES_DETAILS_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    status = data.get("status", "")
    if status == "OK":
        return data.get("result")
    if status in ("NOT_FOUND", "ZERO_RESULTS"):
        return None
    msg = data.get("error_message") or status or "UNKNOWN"
    raise GeocodeError(f"Places Details error: {msg}")


# Optional, limited fallback to classic geocoder if Places fails
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.8, min=1, max=6),
    retry=retry_if_exception_type((requests.RequestException, GeocodeError)),
)
def geocode_google(query: str, api_key: str, region_bias: str = "") -> Optional[Dict[str, Any]]:
    if not api_key:
        raise GeocodeError("Missing GOOGLE_MAPS_API_KEY")
    params = {"address": query, "key": api_key}
    if region_bias:
        params["region"] = region_bias
    resp = requests.get(GEOCODE_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "")
    if status == "OK":
        r0 = data["results"][0]
        loc = r0["geometry"]["location"]
        return {
            "lat": float(loc["lat"]),
            "lon": float(loc["lng"]),
            "formatted_address": r0.get("formatted_address"),
            "provider": "google_geocoding",
            "confidence": r0.get("geometry", {}).get("location_type"),
            "place_id": r0.get("place_id"),
            "types": r0.get("types", []),
            "partial_match": r0.get("partial_match", False),
            "raw": r0,
        }
    if status in ("ZERO_RESULTS",):
        return None
    if status in ("OVER_QUERY_LIMIT", "REQUEST_DENIED", "INVALID_REQUEST", "UNKNOWN_ERROR"):
        msg = data.get("error_message") or status
        raise GeocodeError(f"Google Geocoding error: {msg}")
    msg = data.get("error_message") or status or "UNKNOWN"
    raise GeocodeError(f"Google Geocoding error: {msg}")


# -----------------------
# Selection & acceptance rules
# -----------------------

BRAND_HINTS = [
    # minimal — adjust as needed
    "Hilton", "DoubleTree", "Curio", "Tapestry", "Conrad", "Waldorf",
    "Hilton Garden Inn", "Hampton", "Embassy Suites", "Homewood Suites",
    "Home2 Suites", "Motto", "Tempo", "Signia", "Canopy", "Tru"
]


def _is_lodging_type(types: List[str]) -> bool:
    t = {str(x).lower() for x in (types or [])}
    return "lodging" in t or "premise" in t or "establishment" in t


def _brand_in_name(name: str, brand: str) -> bool:
    name_l = (name or "").lower()
    b = (brand or "").lower()
    if not name_l:
        return False
    # direct brand token
    if b and b in name_l:
        return True
    # any Hilton family hint
    return any(h.lower() in name_l for h in BRAND_HINTS)


def choose_best_place(results: List[Dict[str, Any]], brand: str) -> Optional[Dict[str, Any]]:
    if not results:
        return None
    # Rank: (lodging priority, brand-in-name, rating presence, user_ratings_total)
    ranked = []
    for r in results:
        t = r.get("types", [])
        lodging = 1 if _is_lodging_type(t) else 0
        brand_hit = 1 if _brand_in_name(r.get("name"), brand) else 0
        rating = 1 if r.get("rating") is not None else 0
        ur = int(r.get("user_ratings_total") or 0)
        score = (lodging, brand_hit, rating, ur)
        ranked.append((score, r))
    ranked.sort(key=lambda x: x[0], reverse=True)
    best = ranked[0][1]
    # Require lodging if anything in the set is lodging
    if any(_is_lodging_type(r.get("types", [])) for _, r in ranked) and not _is_lodging_type(best.get("types", [])):
        # pick first lodging one
        for _, r in ranked:
            if _is_lodging_type(r.get("types", [])):
                best = r
                break
    return best


def acceptable_from_details(details: Dict[str, Any]) -> bool:
    if not details:
        return False
    if details.get("business_status") == "CLOSED_PERMANENTLY":
        # Still could be a real rooftop, but generally not desired for active lists
        pass
    if not details.get("geometry"):
        return False
    if not _is_lodging_type(details.get("types", [])):
        # Allow premises if clearly a property
        return False
    return True


# -----------------------
# Orchestration
# -----------------------

def geocode_places_first(queries: List[str], brand: str, base_delay: float) -> Tuple[Optional[Dict[str, Any]], str]:
    last = "NO_RESULT"
    for q in queries:
        try:
            ts = places_text_search(q, API_KEY)
            cand = choose_best_place(ts.get("results", []), brand)
            if cand:
                det = places_details(cand["place_id"], API_KEY)
                if det and acceptable_from_details(det):
                    loc = det["geometry"]["location"]
                    return {
                        "lat": float(loc["lat"]),
                        "lon": float(loc["lng"]),
                        "formatted_address": det.get("formatted_address"),
                        "provider": "google_places",
                        "confidence": "PLACE_DETAILS",  # explicit
                        "place_id": det.get("place_id"),
                        "types": det.get("types", []),
                        "partial_match": False,
                        "raw": det,
                    }, f"PLACES:{q}"
            # limited fallback to classic geocoder for this query
            geo = geocode_google(q + " Hotel", API_KEY, REGION_BIAS)
            if geo and geo.get("confidence", "").upper() == "ROOFTOP" and _is_lodging_type(geo.get("types", [])):
                return geo, f"GEOCODE:{q}"
            last = f"WEAK_OR_EMPTY:{q}"
        except (GeocodeError, requests.RequestException) as e:
            last = f"ERR:{e}"
        finally:
            if base_delay > 0:
                time.sleep(base_delay + random.uniform(0, base_delay * 0.25))
    return None, last


# -----------------------
# Main
# -----------------------

def load_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if not API_KEY:
        raise SystemExit("Set GOOGLE_MAPS_API_KEY env var (or put it in .env).")

    # Load input
    df = pd.read_csv(INPUT_PATH)
    required = {"hotel_name", "group_label"}
    if not required.issubset(df.columns):
        raise SystemExit("Input must contain columns: hotel_name, group_label")
    if "hotel_url" not in df.columns:
        df["hotel_url"] = None

    # Resume support
    already = set()
    if ONLY_MISSING and Path(OUTPUT_PATH).exists():
        prev = pd.read_csv(OUTPUT_PATH)
        if {"hotel_name", "group_label", "lat", "lon"}.issubset(prev.columns):
            for _, r in prev.dropna(subset=["lat", "lon"]).iterrows():
                already.add((str(r["hotel_name"]).strip().lower(), str(r["group_label"]).strip().lower()))

    cache_path = Path(CACHE_PATH)
    cache = load_cache(cache_path)

    work = df.copy()
    if ONLY_MISSING:
        mask = ~work.apply(lambda r: (str(r["hotel_name"]).strip().lower(),
                                      str(r["group_label"]).strip().lower()) in already, axis=1)
        work = work[mask]
    if OFFSET:
        work = work.iloc[OFFSET:]
    if MAX_ROWS and MAX_ROWS > 0:
        work = work.iloc[:MAX_ROWS]

    rows = []
    total = len(work)
    for idx, row in enumerate(work.itertuples(index=False), 1):
        hotel = str(getattr(row, "hotel_name")).strip()
        brand = str(getattr(row, "group_label")).strip()
        group_type = str(getattr(row, "group_type", "Brand")).strip() or "Brand"
        hotel_url = str(getattr(row, "hotel_url", "") or "").strip()

        queries = build_queries(hotel, brand, hotel_url)
        cache_key = json.dumps({
            "v": CACHE_VERSION,
            "provider": "google_places_first",
            "queries": queries,
            "brand": brand,
        }, sort_keys=True)

        cached = cache.get(cache_key, None)
        if cached is not None:
            res = None if cached == {"status": "NO_RESULT"} else cached
            status = "CACHED" if res else "CACHED:NO_RESULT"
        else:
            res, status = geocode_places_first(queries, brand, BASE_DELAY)
            cache[cache_key] = res or {"status": "NO_RESULT"}
            save_cache(cache_path, cache)

        out = {
            "hotel_name": hotel,
            "hotel_url": hotel_url or None,
            "group_label": brand,
            "group_type": group_type,
            "lat": None,
            "lon": None,
            "formatted_address": None,
            "provider": "google_places_first",
            "confidence": None,
            "place_id": None,
            "types": None,
            "partial_match": None,
            "status": status if res else f"NO_RESULT:{status}",
        }
        if res:
            types_out = res.get("types", [])
            out.update({
                "lat": res.get("lat"),
                "lon": res.get("lon"),
                "formatted_address": res.get("formatted_address"),
                "provider": res.get("provider", "google_places"),
                "confidence": res.get("confidence"),
                "place_id": res.get("place_id"),
                "types": ";".join(types_out) if isinstance(types_out, list) else types_out,
                "partial_match": bool(res.get("partial_match", False)),
            })

        rows.append(out)
        print(f"[{idx}/{total}] {hotel} -> {out['status']}")

    out_df = pd.DataFrame(rows, columns=[
        "hotel_name","hotel_url","group_label","group_type",
        "lat","lon","formatted_address","provider","confidence",
        "place_id","types","partial_match","status"
    ])

    if ONLY_MISSING and Path(OUTPUT_PATH).exists():
        prev = pd.read_csv(OUTPUT_PATH)
        keycols = ["hotel_name","group_label"]
        merged = pd.concat([prev, out_df]).drop_duplicates(subset=keycols, keep="last")
        merged.to_csv(OUTPUT_PATH, index=False)
        print(f"Wrote {len(merged)} rows -> {OUTPUT_PATH}")
    else:
        out_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Wrote {len(out_df)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
