import csv
import re
import unicodedata
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import HTTPCookieProcessor, Request, build_opener

URL_TEMPLATE = "https://www.americanexpress.com/en-us/travel/discover/property-results/r/{page}"
OUT = "fhr_thc_hotels.csv"


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("®", "").replace("™", "")
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


class AmexCardHTMLParser(HTMLParser):
    """Parse AMEX property results cards from server-rendered HTML."""

    def __init__(self):
        super().__init__()
        self.rows = []

        self._stack = []
        self._capture_program = False
        self._capture_brand = False
        self._capture_location = False
        self._capture_supplier = False

        self._program_buf = []
        self._brand_buf = []
        self._location_buf = []
        self._supplier_buf = []

        self._current_href = ""
        self._last_program = ""
        self._last_brand = ""
        self._last_location = ""

    @staticmethod
    def _class_has(attrs, needle):
        classes = (attrs.get("class", "") or "").split()
        return needle in classes

    def _pop_until_matching_tag(self, tag):
        """Pop stack entries until we find a matching opening tag.

        HTML from upstream can be imperfect. Using strict LIFO matching can
        desynchronize parsing state and drop rows.
        """
        while self._stack:
            started_tag, started_attrs = self._stack.pop()
            if started_tag == tag:
                return started_tag, started_attrs
        return None, None

    def handle_starttag(self, tag, attrs_list):
        attrs = dict(attrs_list)
        self._stack.append((tag, attrs))

        if tag == "div" and self._class_has(attrs, "card-program"):
            self._capture_program = True
            self._program_buf = []

        if tag == "div" and self._class_has(attrs, "card-brand"):
            self._capture_brand = True
            self._brand_buf = []

        if tag == "div" and self._class_has(attrs, "card-location"):
            self._capture_location = True
            self._location_buf = []

        if tag == "a":
            cls = attrs.get("class", "")
            href = attrs.get("href", "")
            if "card-supplierName" in cls or "/travel/discover/property/" in href:
                self._capture_supplier = True
                self._supplier_buf = []
                self._current_href = href

    def handle_endtag(self, tag):
        started_tag, started_attrs = self._pop_until_matching_tag(tag)
        if not started_tag:
            return
        if tag == "div" and started_tag == "div":
            if self._class_has(started_attrs, "card-program"):
                self._capture_program = False
                self._last_program = _clean_text("".join(self._program_buf))
            if self._class_has(started_attrs, "card-brand"):
                self._capture_brand = False
                self._last_brand = _clean_text("".join(self._brand_buf))
            if self._class_has(started_attrs, "card-location"):
                self._capture_location = False
                self._last_location = _clean_text("".join(self._location_buf))

        if tag == "a" and started_tag == "a" and self._capture_supplier:
            self._capture_supplier = False
            hotel_name = _clean_text("".join(self._supplier_buf))
            href = (self._current_href or "").strip()
            if hotel_name and href and self._last_program:
                self.rows.append(
                    {
                        "program": self._last_program,
                        "brand": self._last_brand,
                        "location": self._last_location,
                        "hotel_name": hotel_name,
                        "hotel_url": href,
                    }
                )
            self._supplier_buf = []
            self._current_href = ""

    def handle_data(self, data):
        if self._capture_program:
            self._program_buf.append(data)
        if self._capture_brand:
            self._brand_buf.append(data)
        if self._capture_location:
            self._location_buf.append(data)
        if self._capture_supplier:
            self._supplier_buf.append(data)


def _dumb_fetch_html(url, opener, timeout=45):
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
            "Referer": "https://www.americanexpress.com/en-us/travel/",
            "Upgrade-Insecure-Requests": "1",
        },
    )
    with opener.open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _normalize_rows(rows, base_url):
    cleaned = []
    for row in rows:
        program = _clean_text(row.get("program", ""))
        brand = _clean_text(row.get("brand", ""))
        location = _clean_text(row.get("location", ""))
        hotel_name = _clean_text(row.get("hotel_name", ""))
        hotel_url = (row.get("hotel_url", "") or "").strip()

        if not program or not hotel_name or not hotel_url:
            continue

        cleaned.append(
            {
                "program_label": program,
                "brand_label": brand,
                "hotel_location": location,
                "hotel_name": hotel_name,
                "hotel_url": urljoin(base_url, hotel_url),
                "group_label": program,
                "group_type": "Program",
            }
        )
    return cleaned


def scrape_all_pages(max_pages=50):
    opener = build_opener(HTTPCookieProcessor())
    all_rows = []
    empty_pages = 0

    try:
        _dumb_fetch_html("https://www.americanexpress.com/", opener, timeout=30)
    except Exception:
        pass

    for n in range(1, max_pages + 1):
        url = URL_TEMPLATE.format(page=n)
        try:
            html = _dumb_fetch_html(url, opener)
        except (HTTPError, URLError, TimeoutError):
            empty_pages += 1
            if empty_pages >= 2:
                break
            continue

        blocked = "Access Denied" in html or "Request unsuccessful" in html
        if blocked or len(html) < 1500:
            empty_pages += 1
            if empty_pages >= 2:
                break
            continue

        parser = AmexCardHTMLParser()
        parser.feed(html)
        rows = _normalize_rows(parser.rows, url)

        if not rows:
            empty_pages += 1
            if empty_pages >= 2:
                break
            continue

        empty_pages = 0
        all_rows.extend(rows)

    dedup = {}
    for r in all_rows:
        key = (
            r["hotel_name"].lower(),
            r["brand_label"].lower(),
            r["program_label"].lower(),
            r["hotel_location"].lower(),
        )
        dedup.setdefault(key, r)

    return list(dedup.values())


def write_output(rows):
    cols = [
        "program_label",
        "brand_label",
        "hotel_location",
        "hotel_name",
        "hotel_url",
        "group_label",
        "group_type",
    ]

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            (r.get("program_label") or "").lower(),
            (r.get("brand_label") or "").lower(),
            (r.get("hotel_location") or "").lower(),
            (r.get("hotel_name") or "").lower(),
        ),
    )

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({c: row.get(c, "") for c in cols})

    print(f"Wrote {len(sorted_rows)} rows -> {OUT} (source=dumb-fetch)")


def main():
    rows = scrape_all_pages()
    if not rows:
        raise RuntimeError("Failed to fetch AMEX pages or no rows were parsed.")
    write_output(rows)


if __name__ == "__main__":
    main()
