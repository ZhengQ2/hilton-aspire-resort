import json
import os
import re
import sys
import unicodedata
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import HTTPCookieProcessor, Request, build_opener

import pandas as pd

URL = "https://www.hilton.com/en/p/hilton-honors/resort-credit-eligible-hotels/"
CACHE_DIR = os.environ.get("HILTON_CACHE_DIR", "cache")
OUT = os.path.join(CACHE_DIR, "hilton_resort_credit_hotels_by_brand.csv")
CACHE_FILE = os.path.join(CACHE_DIR, "geocode_cache_google.json")
NORMALIZED_URL_LOG = os.path.join(CACHE_DIR, "hilton_normalized_urls.log")
LEGACY_CACHE_FILE = "geocode_cache_google.json"

SPECIAL_URL_MAP = {
    "https://romecavalieri.com/": "https://www.hilton.com/en/hotels/romhiwa-rome-cavalieri/",
    "https://www.grandwailea.com/": "https://www.hilton.com/en/hotels/jhmgwwa-grand-wailea/",
    "https://www.waldorfastoriamonarchbeach.com/": "https://www.hilton.com/en/hotels/snamowa-waldorf-astoria-monarch-beach/resort/",
}


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("®", "").replace("™", "")
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    if u.startswith(("javascript:", "#")):
        return ""
    try:
        p = urlparse(u)
        qs = [
            (k, v)
            for k, v in parse_qsl(p.query)
            if k.lower()
            not in {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
                "gclid",
                "mc_cid",
                "mc_eid",
            }
        ]
        p = p._replace(query=urlencode(qs))
        return urlunparse(p)
    except Exception:
        return u


def _normalize_hilton_url(u: str) -> Tuple[str, bool]:
    cleaned = _clean_url(u)
    if not cleaned:
        return "", False

    lower = cleaned.lower()
    for source, target in SPECIAL_URL_MAP.items():
        if lower == source.lower():
            return target, False

    parsed = urlparse(cleaned)
    path_parts = [p for p in parsed.path.split("/") if p]
    if parsed.netloc.lower() == "www.hilton.com" and len(path_parts) >= 2 and path_parts[1].lower() == "hotels":
        lang = path_parts[0].lower()
        if lang != "en":
            path_parts[0] = "en"
            parsed = parsed._replace(path="/" + "/".join(path_parts) + ("/" if parsed.path.endswith("/") else ""))
            return urlunparse(parsed), False
        return cleaned, False

    print(
        f"WARNING: URL does not start with https://www.hilton.com/(language)/hotels: {u}",
        file=sys.stderr,
    )
    return u, True


class HiltonBrandParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: List[Dict[str, str]] = []
        self._stack: List[Tuple[str, Dict[str, str]]] = []

        self._capture_tab = False
        self._tab_buf: List[str] = []
        self._tab_target = ""
        self._panel_labels: Dict[str, str] = {}
        self._brand_stack: List[str] = []

        self._capture_anchor = False
        self._anchor_buf: List[str] = []
        self._anchor_href = ""

    @staticmethod
    def _attrs(attrs_list):
        return {k: v for k, v in attrs_list}

    def handle_starttag(self, tag, attrs_list):
        attrs = self._attrs(attrs_list)
        self._stack.append((tag, attrs))

        if tag == "button" and attrs.get("role") == "tab" and attrs.get("aria-controls"):
            self._capture_tab = True
            self._tab_target = attrs.get("aria-controls", "")
            self._tab_buf = []

        tag_id = attrs.get("id", "")
        if tag_id and tag_id in self._panel_labels:
            self._brand_stack.append(self._panel_labels[tag_id])

        if tag == "a":
            href = (attrs.get("href") or "").strip()
            if href and not href.startswith(("#", "javascript:")):
                self._capture_anchor = True
                self._anchor_href = href
                self._anchor_buf = []

    def handle_endtag(self, tag):
        if not self._stack:
            return
        start_tag, attrs = self._stack.pop()
        if start_tag != tag:
            return

        if tag == "button" and self._capture_tab:
            label = _clean_text("".join(self._tab_buf))
            if label and self._tab_target:
                self._panel_labels[self._tab_target] = label
            self._capture_tab = False
            self._tab_buf = []
            self._tab_target = ""

        tag_id = attrs.get("id", "")
        if tag_id and tag_id in self._panel_labels and self._brand_stack:
            self._brand_stack.pop()

        if tag == "a" and self._capture_anchor:
            name = _clean_text("".join(self._anchor_buf))
            brand = self._brand_stack[-1] if self._brand_stack else ""
            if name and self._anchor_href:
                self.rows.append({"hotel_name": name, "hotel_url": self._anchor_href, "group_label": brand})
            self._capture_anchor = False
            self._anchor_href = ""
            self._anchor_buf = []

    def handle_data(self, data):
        if self._capture_tab:
            self._tab_buf.append(data)
        if self._capture_anchor:
            self._anchor_buf.append(data)


def _dumb_fetch_html(url: str, opener, timeout: int = 60) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Upgrade-Insecure-Requests": "1",
        },
    )
    with opener.open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _collect_hotels_dumb_fetch() -> List[Dict[str, Any]]:
    opener = build_opener(HTTPCookieProcessor())
    html = _dumb_fetch_html(URL, opener)
    parser = HiltonBrandParser()
    parser.feed(html)

    rows = []
    for item in parser.rows:
        name = _clean_text(item.get("hotel_name", ""))
        href = _clean_url(item.get("hotel_url", ""))
        brand = _clean_text(item.get("group_label", "")) or "Unknown"
        if not name or not href:
            continue
        if not urlparse(href).scheme:
            href = urljoin(URL, href)

        normalized_href, has_warning = _normalize_hilton_url(href)
        if not normalized_href:
            continue

        if "/hotels/" not in normalized_href.lower() and "hilton.com" not in normalized_href.lower():
            continue

        rows.append(
            {
                "hotel_name": name,
                "hotel_location": "",
                "hotel_url": normalized_href,
                "original_hotel_url": href,
                "url_warning": has_warning,
                "group_label": brand,
                "group_type": "Brand",
            }
        )

    dedup = {}
    for r in rows:
        key = (r["hotel_name"].lower(), r["group_label"].lower())
        dedup.setdefault(key, r)
    return list(dedup.values())


class HotelAddressParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._capture = False
        self._buf: List[str] = []
        self.location = ""

    @staticmethod
    def _attrs(attrs_list):
        return {k: v for k, v in attrs_list}

    def handle_starttag(self, tag, attrs_list):
        if self.location:
            return
        attrs = self._attrs(attrs_list)
        if tag == "a":
            href = attrs.get("href", "")
            if "google.com/maps/search/?api=1" in href:
                self._capture = True
                self._buf = []

    def handle_endtag(self, tag):
        if tag == "a" and self._capture:
            text = _clean_text("".join(self._buf))
            if text:
                self.location = text
            self._capture = False
            self._buf = []

    def handle_data(self, data):
        if self._capture:
            self._buf.append(data)


def _fetch_hotel_locations(rows: List[Dict[str, Any]]) -> int:
    opener = build_opener(HTTPCookieProcessor())
    fetched = 0
    for row in rows:
        url = row.get("hotel_url", "")
        if not url:
            continue
        try:
            html = _dumb_fetch_html(url, opener, timeout=45)
            parser = HotelAddressParser()
            parser.feed(html)
            row["hotel_location"] = parser.location
            fetched += 1
        except Exception:
            row["hotel_location"] = row.get("hotel_location", "")
    return fetched


def _names_match(name1: str, name2: str) -> bool:
    return name1 in name2 or name2 in name1


def _load_geocode_cache_index(cache_path: str = CACHE_FILE):
    source_path = cache_path
    if not os.path.exists(source_path) and os.path.exists(LEGACY_CACHE_FILE):
        source_path = LEGACY_CACHE_FILE
    if not os.path.exists(source_path):
        return {}, set()

    with open(source_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    index = {}
    cached_hotels = set()
    for key in cache.keys():
        try:
            meta = json.loads(key)
        except Exception:
            continue

        brand = _clean_text(meta.get("brand", "")).lower()
        if not brand:
            continue
        queries_raw = meta.get("queries", []) or []
        queries_clean = [_clean_text(q) for q in queries_raw if q]
        queries_lower = [q.lower() for q in queries_clean if q]
        if not queries_lower:
            continue

        index.setdefault(brand, set()).update(queries_lower)
        canonical_name = queries_lower[-1]
        if canonical_name:
            cached_hotels.add((canonical_name, brand))

    return index, cached_hotels


def _is_hotel_in_cache(hotel_name: str, brand: str, cache_index) -> bool:
    brand_key = _clean_text(brand).lower()
    queries = cache_index.get(brand_key)
    if not queries:
        return False
    name = _clean_text(hotel_name).lower()
    return any(_names_match(name, q) for q in queries)


def _is_cached_hotel_in_scraped(cached_name: str, brand: str, scraped_index) -> bool:
    brand_key = _clean_text(brand).lower()
    name = _clean_text(cached_name).lower()
    scraped_names = scraped_index.get(brand_key, [])
    return any(_names_match(name, s) for s in scraped_names)


def _read_cache_dict(cache_path: str) -> Dict[str, Any]:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(LEGACY_CACHE_FILE):
        with open(LEGACY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _update_cache_with_new_hotels(cache_path: str, new_hotels: List[Dict[str, Any]]) -> int:
    if not new_hotels:
        return 0
    try:
        cache = _read_cache_dict(cache_path)
        added_count = 0
        for hotel_data in new_hotels:
            hotel_name = _clean_text(hotel_data["hotel_name"])
            brand = _clean_text(hotel_data["group_label"])
            cache_key = json.dumps(
                {
                    "brand": brand,
                    "provider": "google_places_first",
                    "queries": [hotel_name],
                    "v": 3,
                },
                sort_keys=True,
            )
            if cache_key not in cache:
                cache[cache_key] = None
                added_count += 1

        if added_count > 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        return added_count
    except Exception as e:
        print(f"ERROR: Failed to update cache: {e}")
        return 0


def _remove_hotels_from_cache(cache_path: str, removed_hotels: List[Tuple[str, str]], scraped_index: Dict[str, List[str]]) -> int:
    if not removed_hotels:
        return 0
    try:
        cache = _read_cache_dict(cache_path)
        keys_to_remove = []
        for key in list(cache.keys()):
            try:
                meta = json.loads(key)
            except Exception:
                continue
            brand = _clean_text(meta.get("brand", "")).lower()
            queries = meta.get("queries", [])
            if not queries:
                continue
            canonical_name = _clean_text(queries[-1]).lower()
            for removed_name, removed_brand in removed_hotels:
                if brand != _clean_text(removed_brand).lower():
                    continue
                if _names_match(canonical_name, _clean_text(removed_name).lower()) and not _is_cached_hotel_in_scraped(canonical_name, brand, scraped_index):
                    keys_to_remove.append(key)
                    break

        removed_count = 0
        for key in keys_to_remove:
            if key in cache:
                del cache[key]
                removed_count += 1

        if removed_count > 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        return removed_count
    except Exception as e:
        print(f"ERROR: Failed to remove hotels from cache: {e}")
        return 0


def main() -> None:
    _ensure_cache_dir()

    try:
        rows = _collect_hotels_dumb_fetch()
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Failed to fetch Hilton hotels list: {e}")

    with open(NORMALIZED_URL_LOG, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['hotel_name']} -> {r['hotel_url']}\n")

    location_count = _fetch_hotel_locations(rows)
    print(f"Fetched hotel locations for {location_count} URLs")

    df = pd.DataFrame(
        rows,
        columns=[
            "hotel_name",
            "hotel_location",
            "hotel_url",
            "original_hotel_url",
            "url_warning",
            "group_label",
            "group_type",
        ],
    ).sort_values(by=["group_label", "hotel_name"])
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows -> {OUT}")

    cache_index, cached_hotels = _load_geocode_cache_index()
    if not cache_index:
        print(f"No cache index built (file missing or empty: {CACHE_FILE}).")
    else:
        scraped_index: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            bkey = _clean_text(row["group_label"]).lower()
            nkey = _clean_text(row["hotel_name"]).lower()
            scraped_index.setdefault(bkey, []).append(nkey)

        new_hotels_mask = df.apply(
            lambda row: not _is_hotel_in_cache(row["hotel_name"], row["group_label"], cache_index), axis=1
        )
        new_hotels = df[new_hotels_mask].to_dict("records")
        if not new_hotels:
            print("All scraped hotels appear to be present in geocode cache.")
        else:
            print("Hotels NOT found in geocode cache (new hotels):")
            for r in new_hotels:
                print(f"- {r['hotel_name']}  [brand: {r['group_label']}]  -> {r['hotel_url']}")

        removed_hotels = []
        for cached_name, brand in cached_hotels:
            if not _is_cached_hotel_in_scraped(cached_name, brand, scraped_index):
                removed_hotels.append((cached_name, brand))

        if not removed_hotels:
            print("No cached hotels appear to have been removed from the current list.")
        else:
            print("Hotels in geocode cache but NOT in current scraped list (removed):")
            for name, brand in sorted(removed_hotels, key=lambda x: (x[1], x[0])):
                print(f"- {name}  [brand: {brand}]")

        if new_hotels:
            print(f"\nUpdating cache with {len(new_hotels)} new hotels...")
            added = _update_cache_with_new_hotels(CACHE_FILE, new_hotels)
            if added > 0:
                print(f"✓ Added {added} new hotel(s) to cache")

        if removed_hotels:
            print(f"\nRemoving {len(removed_hotels)} hotels from cache...")
            removed = _remove_hotels_from_cache(CACHE_FILE, removed_hotels, scraped_index)
            if removed > 0:
                print(f"✓ Removed {removed} hotel(s) from cache")


if __name__ == "__main__":
    main()
