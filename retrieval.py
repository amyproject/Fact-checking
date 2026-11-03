"""Evidence retrieval with two backends and leakage guards.

wikipedia: free, no key. Current article text, so it can post-date the video.
brave:     web search capped to results published on or before the video date.

Both drop fact-check sites (they publish the answer key) and the claim's own
fact-check article / original post.
"""
import hashlib
import html
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime

from common import CACHE

RETRIEVER = os.environ.get("RETRIEVER", "wikipedia")
# Wikimedia throttles anonymous agents; its API policy asks for a contact URL in the UA
UA = {"User-Agent": "shortvideo-factcheck/0.1 (research benchmark; https://www.mediawiki.org/wiki/API:Etiquette)"}

BLOCK_DOMAINS = (
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "africacheck.org",
    "leadstories.com", "checkyourfact.com", "truthorfiction.com", "boomlive.in", "altnews.in",
    "factly.in", "factcheck.afp.com", "healthfeedback.org", "sciencefeedback.co",
    "climatefeedback.org", "newschecker.in", "vishvasnews.com", "pesacheck.org",
    "dubawa.org", "logically.ai", "misbar.com", "maldita.es", "aosfatos.org", "teyit.org",
    "verafiles.org", "rappler.com/newsbreak/fact-check", "mediabiasfactcheck.com",
)
BLOCK_PATH_HINTS = ("fact-check", "factcheck", "fact_check", "hoax-slayer", "/verify/")


def blocked(url, extra=()):
    u = url.lower()
    host = urllib.parse.urlparse(u).netloc.removeprefix("www.")
    if any(host == d or host.endswith("." + d) or u.split("://", 1)[-1].startswith(d) for d in BLOCK_DOMAINS):
        return True
    if any(h in u for h in BLOCK_PATH_HINTS):
        return True
    return any(e and _norm_url(e) == _norm_url(url) for e in extra)


def _norm_url(u):
    u = re.sub(r"^.*?web\.archive\.org/web/\d+\w*/", "", u.strip().lower())
    return re.sub(r"^https?://(www\.)?", "", u).rstrip("/")


_HTTP_SLOTS = threading.Semaphore(2)  # Wikipedia and Brave both 429 under bursty parallel calls


def _get_json(url, headers=None):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    for attempt in range(6):
        try:
            with _HTTP_SLOTS, urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(2 ** attempt)


def _cached(kind, key, fn):
    f = CACHE / "search" / kind / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text())
    out = fn()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


_WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to in on for and or is are was were be been it its this that with as at by from "
           "he she they we you i his her their our not no but has have had will would can could".split())


def tokens(text):
    return [w for w in _WORD.findall(text.lower()) if w not in STOP and (len(w) > 2 or w.isdigit())]


def _best_passages(text, query, n=2, words=120):
    q = set(tokens(query))
    paras = [p for p in text.split("\n") if len(p.split()) > 8]
    chunks = []
    for p in paras:
        w = p.split()
        chunks += [" ".join(w[i:i + words]) for i in range(0, len(w), words)]
    scored = sorted(chunks, key=lambda c: -len(q & set(tokens(c))))
    return scored[:n]


def _wikipedia(query, k):
    def run():
        api = "https://en.wikipedia.org/w/api.php?"

        def titles(q):
            hits = _get_json(api + urllib.parse.urlencode(
                {"action": "query", "list": "search", "srsearch": q, "srlimit": k, "format": "json"}))
            return hits.get("query", {}).get("search", [])
        # Wikipedia search ANDs every word, so a sentence-long query often finds nothing;
        # fall back to its longest (usually rarest) 5, then 3, content words
        words = sorted(set(tokens(query)), key=len, reverse=True)
        found = titles(query) or titles(" ".join(words[:5])) or titles(" ".join(words[:3]))
        out = []
        for h in found:
            title = h["title"]
            page = _get_json(api + urllib.parse.urlencode(
                {"action": "query", "prop": "extracts", "explaintext": 1, "titles": title,
                 "format": "json", "redirects": 1}))
            text = next(iter(page["query"]["pages"].values())).get("extract", "")
            url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
            for p in _best_passages(text, query):
                out.append({"url": url, "title": title, "text": p, "date": None})
        return out
    return _cached("wikipedia", f"{query}|{k}", run)


def _brave(query, k, before):
    key = os.environ["BRAVE_API_KEY"]
    params = {"q": query, "count": max(k * 2, 10)}
    if before:
        params["freshness"] = f"1990-01-01to{before}"

    def run():
        res = _get_json("https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(params),
                        headers={"Accept": "application/json", "X-Subscription-Token": key})
        out = []
        for r in res.get("web", {}).get("results", []):
            text = " ".join([r.get("description", "")] + r.get("extra_snippets", []))
            text = html.unescape(re.sub(r"<[^>]+>", "", text))
            out.append({"url": r["url"], "title": r.get("title", ""), "text": text, "date": r.get("page_age")})
        return out
    return _cached("brave", json.dumps(params, sort_keys=True), run)


def parse_claim_date(s):
    """AVeriTeC dates are d-m-yyyy. Returns yyyy-mm-dd or None."""
    try:
        return datetime.strptime(s, "%d-%m-%Y").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def search(query, k=3, before=None, exclude=()):
    """Top passages for a query, minus blocked sources. `before` is yyyy-mm-dd."""
    try:
        hits = _brave(query, k, before) if RETRIEVER == "brave" else _wikipedia(query, k)
    except Exception as e:  # a failed search means less evidence, not a crashed run
        print(f"  search failed ({RETRIEVER}): {query[:60]!r}: {e}")
        return []
    return [h for h in hits if h["text"] and not blocked(h["url"], exclude)][: k * 2]
