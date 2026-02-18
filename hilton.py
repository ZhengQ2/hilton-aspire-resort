# scrape_hilton_hotels_by_brand_playwright.py
from playwright.sync_api import sync_playwright, TimeoutError
import pandas as pd
import re
import unicodedata
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from typing import Dict, List, Tuple, Any, Optional
import json
import os

URL = "https://www.hilton.com/en/p/hilton-honors/resort-credit-eligible-hotels/"
OUT = "cache/hilton_hotels.csv"
CACHE_FILE = "cache/geocode_cache_google_hilton.json"


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
    # Normalize & drop obvious tracking params (keep it light)
    try:
        p = urlparse(u)
        if not p.scheme:
            # leave as-is; caller may join against page URL later
            pass
        # remove some common trackers
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


def _visible_text(el):
    return el.evaluate("n => (n.innerText || '').replace(/\\s+/g,' ').trim()")


def _click_all_show_more(panel):
    more_selectors = [
        "button:has-text('Show more')",
        "button:has-text('Show More')",
        "button:has-text('VIEW MORE')",
        "button:has-text('View more')",
        "a:has-text('Show more')",
        "a:has-text('View more')",
    ]
    changed = True
    attempts = 0
    while changed and attempts < 8:
        changed = False
        attempts += 1
        for sel in more_selectors:
            for b in panel.locator(sel).all():
                try:
                    if b.is_visible():
                        b.click(timeout=1000)
                        changed = True
                except Exception:
                    pass
        if changed:
            panel.page.wait_for_timeout(400)


def _force_lazy_render(panel):
    try:
        panel.evaluate(
            """
        (root) => {
            const el = root;
            if (!el) return;
            el.scrollTop = 0;
        }
        """
        )
    except Exception:
        pass
    for _ in range(10):
        try:
            panel.evaluate("(el) => { el.scrollTop = el.scrollHeight; }")
        except Exception:
            break
        panel.page.wait_for_timeout(150)


def _collect_hotel_links(panel, base_url):
    """
    Return list of dicts: [{name, href}], preferring Hilton property pages.
    """
    js = r"""
    (root) => {
      const rows = [];
      const clean = s => (s || "").replace(/\s+/g, " ").trim().replace(/[®™]/g, "");
      const isJunkText = t => /^(view|book|details?|rates?)\b/i.test(t);
      const abs = (href) => {
        try { return new URL(href, location.href).href; } catch { return href || ""; }
      };

      const push = (name, href) => {
        name = clean(name);
        href = (href || "").trim();
        if (!name || isJunkText(name)) return;
        if (!href || href.startsWith("#") || href.startsWith("javascript:")) return;
        rows.push({ name, href: abs(href) });
      };

      // Primary: anchors that look like property links
      root.querySelectorAll("a").forEach(a => {
        const t = clean(a.textContent || a.getAttribute("aria-label") || "");
        const href = a.getAttribute("href") || "";
        if (!t || t.length > 160) return;
        if (isJunkText(t)) return;
        push(t, href);
      });

      // Fallback: role=link (sometimes divs), try nearest/inner anchor
      root.querySelectorAll("[role='link']").forEach(el => {
        const t = clean(el.textContent || el.getAttribute("aria-label") || "");
        if (!t || t.length > 160 || isJunkText(t)) return;
        const a = el.closest("a") || el.querySelector("a");
        const href = a ? (a.getAttribute("href") || "") : "";
        if (href) push(t, href);
      });

      // Final fallback: cards with one obvious link
      root.querySelectorAll("[data-testid*='card'], [class*='card']").forEach(card => {
        const a = card.querySelector("a");
        if (!a) return;
        const t = clean(a.textContent || a.getAttribute("aria-label") || "");
        const href = a.getAttribute("href") || "";
        if (!t || t.length > 160 || isJunkText(t)) return;
        if (href) push(t, href);
      });

      return rows;
    }
    """
    try:
        items = panel.evaluate(js) or []
    except Exception:
        items = []

    # Clean and normalize
    cleaned = []
    for it in items:
        name = _clean_text(it.get("name") or "")
        href = _clean_url(it.get("href") or "")
        if not name or not href:
            continue
        if not urlparse(href).scheme:
            href = urljoin(base_url, href)
        cleaned.append({"name": name, "href": href})

    # Dedup per name, preferring Hilton property pages when multiple URLs exist
    def score(u: str) -> int:
        u = u.lower()
        # Prefer direct property pages on hilton.com/en/hotels/
        if "hilton.com" in u and "/en/hotels/" in u:
            return 3
        if "hilton.com" in u:
            return 2
        return 1

    by_name = {}
    for row in cleaned:
        key = row["name"].lower()
        best = by_name.get(key)
        if not best or score(row["href"]) > score(best["href"]):
            by_name[key] = row

    return list(by_name.values())


def _grab_from_panel(panel, brand_label, base_url):
    _click_all_show_more(panel)
    _force_lazy_render(panel)
    items = _collect_hotel_links(panel, base_url)
    rows = []
    for it in items:
        rows.append(
            {
                "hotel_name": it["name"],
                "hotel_url": it["href"],
                "group_label": brand_label,
                "group_type": "Brand",
            }
        )
    return rows


def scrape_desktop(page):
    rows = []
    container = page.locator("#HotelsByBrand")
    tablist = container.locator("[role='tablist'] button[role='tab']")
    if tablist.count() == 0:
        return rows
    for i in range(tablist.count()):
        btn = tablist.nth(i)
        label = _clean_text(btn.inner_text())
        panel_id = btn.get_attribute("aria-controls")
        btn.click()
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except TimeoutError:
            pass
        panel = (
            container.locator(f"#{panel_id}")
            if panel_id
            else container.locator("[role='tabpanel']").nth(i)
        )
        try:
            panel.wait_for(state="visible", timeout=7000)
        except TimeoutError:
            pass
        rows.extend(_grab_from_panel(panel, label, page.url))
    return rows


def scrape_mobile(page):
    rows = []
    container = page.locator("#HotelsByBrand")
    triggers = container.locator("[aria-controls^='radix-']")
    if triggers.count() == 0:
        return rows
    for i in range(triggers.count()):
        trig = triggers.nth(i)
        label = _clean_text(trig.inner_text())
        panel_id = trig.get_attribute("aria-controls")
        if not panel_id:
            continue
        panel = container.locator(f"#{panel_id}")
        if (trig.get_attribute("aria-expanded") or "").lower() != "true":
            trig.click()
        try:
            panel.wait_for(state="visible", timeout=7000)
        except TimeoutError:
            pass
        page.wait_for_timeout(300)
        rows.extend(_grab_from_panel(panel, label, page.url))
    return rows


# ---------- NEW: cache helpers ----------

def _load_geocode_cache_index(cache_path: str = CACHE_FILE):
    """
    Load cache/geocode_cache_google_hilton.json and build:
      - index: { brand_lower: set([query_lower, ...]) }
      - cached_hotels: set([(hotel_name_lower, brand_lower), ...])

    For hotel_name, we heuristically take the *last* query as the canonical
    name (matches your example where the final query is plain hotel name).
    """
    if not os.path.exists(cache_path):
        return {}, set()

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    index = {}
    cached_hotels = set()

    for key in cache.keys():
        # keys are JSON-encoded objects like:
        # {"brand": "...", "provider": "...", "queries": [...], "v": 3}
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

        # Build query index for "in cache" tests
        index.setdefault(brand, set()).update(queries_lower)

        # Heuristic canonical hotel name: last query (usually the simplest)
        canonical_name = queries_lower[-1]
        if canonical_name:
            cached_hotels.add((canonical_name, brand))

    return index, cached_hotels


def _is_hotel_in_cache(hotel_name: str, brand: str, cache_index) -> bool:
    """
    Treat a hotel as cached if, for the same brand, any cached query
    contains the hotel name or is contained in it (case-insensitive).
    """
    if not cache_index:
        return False

    brand_key = _clean_text(brand).lower()
    queries = cache_index.get(brand_key)
    if not queries:
        return False

    name = _clean_text(hotel_name).lower()
    for q in queries:
        if _names_match(name, q):
            return True
    return False


def _is_cached_hotel_in_scraped(cached_name: str, brand: str, scraped_index) -> bool:
    """
    Inverse of _is_hotel_in_cache:
    Check if a cached (hotel, brand) appears in the scraped list for that brand.
    """
    brand_key = _clean_text(brand).lower()
    name = _clean_text(cached_name).lower()

    scraped_names = scraped_index.get(brand_key, [])
    for s in scraped_names:
        if _names_match(name, s):
            return True
    return False


def _names_match(name1: str, name2: str) -> bool:
    """
    Check if two hotel names match using fuzzy logic.
    Returns True if one name contains the other or they're equal.
    
    Args:
        name1: First name (lowercase)
        name2: Second name (lowercase)
        
    Returns:
        True if names match, False otherwise
    """
    n1 = _clean_text(name1).lower()
    n2 = _clean_text(name2).lower()

    if not n1 or not n2:
        return False
    if n1 == n2:
        return True

    # Token-aware matching prevents false positives like
    # "hilton dali" vs "hilton dalian" while still allowing
    # benign suffix/prefix differences.
    generic_tokens = {
        "hotel", "resort", "spa", "and", "the", "at", "by", "&",
        "hilton", "inn", "suites", "collection", "club"
    }

    def tokens(s: str) -> set:
        return {
            t
            for t in re.findall(r"[a-z0-9]+", s)
            if t and t not in generic_tokens
        }

    t1 = tokens(n1)
    t2 = tokens(n2)

    if not t1 or not t2:
        return False
    if t1 == t2:
        return True

    # Allow subset matches when there is substantial overlap and
    # the non-overlapping tokens are only generic words.
    overlap = t1 & t2
    smaller = min(len(t1), len(t2))
    return bool(overlap) and len(overlap) >= max(1, smaller - 1)


def _update_cache_with_new_hotels(
    cache_path: str, 
    new_hotels: List[Dict[str, Any]], 
    cache_index: Dict[str, set]
) -> int:
    """
    Update the geocode cache by adding entries for new hotels.
    
    Args:
        cache_path: Path to the cache file
        new_hotels: List of dictionaries with new hotels to add
        cache_index: Existing cache index for checking duplicates
        
    Returns:
        Number of hotels added to cache
    """
    if not new_hotels:
        return 0
    
    try:
        # Load existing cache
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        else:
            cache = {}
        
        added_count = 0
        for hotel_data in new_hotels:
            hotel_name = _clean_text(hotel_data["hotel_name"])
            brand = _clean_text(hotel_data["group_label"])
            
            # Build query list similar to google-convert.py format
            queries = [hotel_name]
            
            # Create cache key matching the existing format
            cache_key = json.dumps({
                "brand": brand,
                "provider": "google_places_first",
                "queries": queries,
                "v": 3
            }, sort_keys=True)
            
            # Add placeholder entry (will be geocoded later)
            # Using None to indicate it needs geocoding
            cache[cache_key] = None
            added_count += 1
        
        # Save updated cache
        if added_count > 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        
        return added_count
        
    except PermissionError:
        print(f"ERROR: Permission denied when trying to write to {cache_path}")
        return 0
    except Exception as e:
        print(f"ERROR: Failed to update cache: {e}")
        return 0


def _remove_hotels_from_cache(
    cache_path: str, 
    removed_hotels: List[Tuple[str, str]], 
    scraped_index: Dict[str, List[str]]
) -> int:
    """
    Remove hotels from the geocode cache that are no longer in the scraped list.
    
    Args:
        cache_path: Path to the cache file
        removed_hotels: List of (hotel_name, brand) tuples to remove
        scraped_index: Index of scraped hotels by brand
        
    Returns:
        Number of hotels removed from cache
    """
    if not removed_hotels:
        return 0
    
    try:
        # Load existing cache
        if not os.path.exists(cache_path):
            return 0
            
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        
        keys_to_remove = []
        
        # Find cache keys that correspond to removed hotels
        for key in cache.keys():
            try:
                meta = json.loads(key)
                brand = _clean_text(meta.get("brand", "")).lower()
                queries = meta.get("queries", [])
                
                if not queries:
                    continue
                
                # Use last query as canonical name (same as _load_geocode_cache_index)
                canonical_name = _clean_text(queries[-1]).lower()
                
                # Check if this hotel should be removed
                for removed_name, removed_brand in removed_hotels:
                    removed_name_clean = _clean_text(removed_name).lower()
                    removed_brand_clean = _clean_text(removed_brand).lower()
                    
                    # Match by brand and name
                    if brand == removed_brand_clean:
                        # Check if names match (using same fuzzy logic)
                        if _names_match(canonical_name, removed_name_clean):
                            # Double check it's not in scraped list
                            if not _is_cached_hotel_in_scraped(canonical_name, brand, scraped_index):
                                keys_to_remove.append(key)
                                break
            except Exception:
                continue
        
        # Remove keys
        removed_count = 0
        for key in keys_to_remove:
            if key in cache:
                del cache[key]
                removed_count += 1
        
        # Save updated cache
        if removed_count > 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        
        return removed_count
        
    except PermissionError:
        print(f"ERROR: Permission denied when trying to write to {cache_path}")
        return 0
    except Exception as e:
        print(f"ERROR: Failed to remove hotels from cache: {e}")
        return 0


# ----------------------------------------


def main(headless=True):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        page.set_default_navigation_timeout(60000)
        page.goto(URL, wait_until="domcontentloaded")

        rows = scrape_desktop(page)
        if not rows:
            rows = scrape_mobile(page)

        # Dedup by (hotel, brand) while preferring the "best" URL picked above
        dedup = {}
        for r in rows:
            key = (r["hotel_name"].lower(), r["group_label"].lower())
            # first one already chosen by _collect_hotel_links' preference; keep it
            dedup.setdefault(key, r)

        df = pd.DataFrame(
            dedup.values(), columns=["hotel_name", "hotel_url", "group_label", "group_type"]
        ).sort_values(by=["group_label", "hotel_name"])

        df.to_csv(OUT, index=False)
        print(f"Wrote {len(df)} rows -> {OUT}")

        # -------- NEW: compare against geocode cache --------
        cache_index, cached_hotels = _load_geocode_cache_index()
        if not cache_index:
            print(f"No cache index built (file missing or empty: {CACHE_FILE}).")
        else:
            # Index scraped hotels by brand for reverse lookup
            scraped_index = {}
            for _, row in df.iterrows():
                bkey = _clean_text(row["group_label"]).lower()
                nkey = _clean_text(row["hotel_name"]).lower()
                scraped_index.setdefault(bkey, []).append(nkey)

            # 1) Scraped hotels that are NOT in geocode cache (new)
            new_hotels_mask = df.apply(
                lambda row: not _is_hotel_in_cache(row["hotel_name"], row["group_label"], cache_index),
                axis=1
            )
            new_hotels = df[new_hotels_mask].to_dict('records')

            if not new_hotels:
                print("All scraped hotels appear to be present in geocode cache.")
            else:
                print("Hotels NOT found in geocode cache (new hotels):")
                for r in new_hotels:
                    print(
                        f"- {r['hotel_name']}  [brand: {r['group_label']}]  -> {r['hotel_url']}"
                    )

            # 2) Cached hotels that are NOT in scraped list (removed)
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
            
            # 3) Update cache with new hotels
            if new_hotels:
                print(f"\nUpdating cache with {len(new_hotels)} new hotels...")
                added = _update_cache_with_new_hotels(CACHE_FILE, new_hotels, cache_index)
                if added > 0:
                    print(f"✓ Added {added} new hotel(s) to cache")
                elif added == 0:
                    print("⚠ Failed to add hotels to cache (check file permissions)")
            
            # 4) Remove hotels from cache that are no longer scraped
            if removed_hotels:
                print(f"\nRemoving {len(removed_hotels)} hotels from cache...")
                removed = _remove_hotels_from_cache(CACHE_FILE, removed_hotels, scraped_index)
                if removed > 0:
                    print(f"✓ Removed {removed} hotel(s) from cache")
                elif removed == 0:
                    print("⚠ Failed to remove hotels from cache (check file permissions)")

        browser.close()


if __name__ == "__main__":
    import sys

    headed = "--headed" in sys.argv
    main(headless=not headed)
