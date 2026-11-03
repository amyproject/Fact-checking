"""Build the v2 test set: 3x the claims, 4-way labels, and a separate dev split for iterating.

    python 11_build_v2.py

Sources
  1. The 65 AVeriTeC claims of v1, unchanged (data/claims.jsonl), relabeled 4-way.
  2. English ClaimReview fact-checks from the Google / Data Commons feed whose claim
     appeared in a YouTube video (data/raw/claimreview/feed.json). Each has the speaker,
     the date and setting, the fact-checker's rating and explanation, and often a t=
     timestamp. Transcripts are fetched and the claim is located exactly as in v1.

Labels (AVeriTeC's four): supported / refuted / conflicting (cherry-picked, missing
context, half true) / not enough evidence. Ratings that don't map cleanly (flip-flop,
"altered" media, free-text ratings) are dropped.

Splits: v1's 65 claims + 130 new = 195 test claims (3x). Further located claims form a
dev split for iterating on harnesses; the test split is run once per frozen system.

Writes data/v2/{claims,clips}.jsonl and data/v2/stats.json.
"""
import importlib
import json
import random
import re
import sys
import time
from collections import Counter

from common import DATA, read_jsonl, write_jsonl

tb = importlib.import_module("01_build_testset")

FEED = DATA / "raw" / "claimreview" / "feed.json"
OUT = DATA / "v2"
SEED = 0
TEST_ADD = {"supported": 31, "not_enough_evidence": 15, "conflicting": 40, "refuted": 44}   # +130
OFFLINE = "--offline" in sys.argv  # only use cached transcripts (YouTube rate-limits this IP)
DEV = {"supported": 15, "not_enough_evidence": 8, "conflicting": 25, "refuted": 30}
YT = re.compile(r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/|embed/)|youtu\.be/)([\w-]{11})")
FOUR = {"Supported": "supported", "Refuted": "refuted", "Not Enough Evidence": "not_enough_evidence",
        "Conflicting Evidence/Cherrypicking": "conflicting"}
TO3 = {"supported": "supported", "refuted": "contradicted", "not_enough_evidence": "unverified", "conflicting": "unverified"}
RATING = [  # first match wins; checked against the lower-cased, stripped rating
    (r"^(true|correct|accurate|real|largely correct|mostly true|verdadeiro|legit)\.?$", "supported"),
    (r"^(half true|needs context|lacks context|missing context|out of context|not the whole story|cherry ?picking|"
     r"cherry-picks|misleading|distorts the facts|two pinocchios|exaggerat\w*|this is exaggerated|mixture|partly (true|false)|"
     r"verdadeiro, mas\.\.\.)\.?$", "conflicting"),
    (r"^(no basis|no evidence|unproven|unsupported|unverified|lacks evidence|unsubstantiated|insufficient evidence)\.?$",
     "not_enough_evidence"),
    (r"^(false|mostly false|fake|pants on fire|three pinocchios|four pinocchios|inaccurate|wrong|incorrect|not legit|"
     r"mostly not legit|false headline|fabricated|falso|baseless|hoax|scam|bogus|not true|untrue)\.?$", "refuted"),
]


def map_rating(r):
    r = re.sub(r"\s+", " ", (r or "").strip().lower())
    return next((lab for pat, lab in RATING if re.match(pat, r)), None)


def english(t):
    t = t or ""
    return len(t) > 15 and sum(c.isascii() for c in t) / len(t) > 0.97 and \
        len(re.findall(r"\b(the|and|is|of|to|a|in|that|was|are)\b", t.lower())) >= 2


def dmy(iso):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", iso or "")
    return f"{int(m.group(3))}-{int(m.group(2))}-{m.group(1)}" if m else None


def claimreview_candidates():
    feed = json.loads(FEED.read_text())
    seen, out, dropped = set(), [], Counter()
    for el in feed["dataFeedElement"]:
        for it in el.get("item") or []:
            ir = it.get("itemReviewed") or {}
            apps = (ir.get("appearance") or []) + ([ir.get("firstAppearance")] if ir.get("firstAppearance") else [])
            urls = [a.get("url", "") if isinstance(a, dict) else str(a) for a in apps]
            hit = next(((u, m.group(1)) for u in urls for m in [YT.search(u or "")] if m), None)
            if not hit:
                continue
            claim = " ".join(str(it.get("claimReviewed") or "").split()).strip("“”\" ")
            if not english(claim):
                continue
            rating = (it.get("reviewRating") or {}).get("alternateName", "")
            lab = map_rating(rating)
            if not lab:
                dropped[f"rating doesn't map: {str(rating).strip().lower()[:24]}"] += 1
                continue
            key = (hit[1], claim.lower()[:80])
            if key in seen:
                continue
            seen.add(key)
            author = (it.get("author") or {}).get("name", "")
            out.append({
                "id": f"cr-{len(out)}", "source": "claimreview", "video_id": hit[1], "appearance": hit[0],
                "claim": claim, "label4": lab, "rating_raw": str(rating).strip(), "fact_checker": author,
                "justification": " ".join(str((it.get("reviewRating") or {}).get("ratingExplanation") or "").split()),
                "fact_checking_article": it.get("url", ""),
                "speaker": ((ir.get("author") or {}).get("name") or "").strip(),
                "setting": ir.get("name") or "",
                "claim_date": dmy(ir.get("datePublished") or it.get("datePublished")),
                "review_date": it.get("datePublished", ""),
            })
    return out, dropped


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = read_jsonl(DATA / "claims.jsonl")
    v1_clips = read_jsonl(DATA / "clips.jsonl")
    v1_videos = {c["video"] for c in v1}

    cands, dropped = claimreview_candidates()
    by_label = {}
    for c in cands:
        if "yt:" + c["video_id"] in v1_videos:
            dropped["video already in v1"] += 1
            continue
        by_label.setdefault(c["label4"], []).append(c)
    print({k: len(v) for k, v in by_label.items()}, "candidates by label")

    want = {k: TEST_ADD[k] + DEV[k] for k in TEST_ADD}
    located, clips, tried = {k: [] for k in want}, [], Counter()
    blocked = 0
    for lab, pool in by_label.items():
        if blocked >= 5:
            break
        # verbatim-quote fact-checkers first: their claims match transcripts best
        rng.shuffle(pool)
        pool.sort(key=lambda c: c["fact_checker"] not in ("PolitiFact", "FactCheck.org", "Washington Post", "The New York Times"))
        for c in pool:
            if len(located[lab]) >= want[lab]:
                break
            tried[lab] += 1
            cached = (DATA / "transcripts" / f"{c['video_id']}.json").exists()
            if OFFLINE and not cached:
                continue
            snips, reason = tb.fetch_transcript(c["video_id"])
            if not cached:
                time.sleep(0.4)  # be gentle: YouTube blocks bursts
            if reason in ("IpBlocked", "RequestBlocked", "VideoUnplayable"):
                blocked += 1
                if blocked >= 5:
                    print("  YouTube is blocking this IP: stopping and writing what was located (rerun later to fill the rest)")
                    break
                continue
            blocked = 0
            if not snips:
                dropped[f"no transcript: {reason}"] += 1
                continue
            duration = snips[-1]["end"]
            center, how, ov = tb.locate(c["claim"], c["appearance"], snips, duration)
            if ov < tb.MIN_OVERLAP:
                dropped["claim not found in transcript"] += 1
                continue
            if duration <= tb.SHORT_VIDEO_MAX:
                start, end = 0.0, duration
            else:
                start, end = max(0.0, center - tb.CLIP_SECONDS / 2), min(duration, center + tb.CLIP_SECONDS / 2)
            clip_id = f"{c['video_id']}-cr{len(clips)}"
            clips.append({"clip_id": clip_id, "video": "yt:" + c["video_id"],
                          "url": f"https://www.youtube.com/watch?v={c['video_id']}&t={int(start)}",
                          "start": round(start, 1), "end": round(end, 1), "video_seconds": round(duration),
                          "claim_date": c["claim_date"], "transcript": tb.window_text(snips, start, end)})
            located[lab].append({**c, "clip_id": clip_id, "video": "yt:" + c["video_id"], "located_by": how,
                                 "overlap": round(ov, 2), "gold": TO3[lab], "gold_raw": c["rating_raw"],
                                 "claim_types": [], "original_claim_url": c["appearance"]})
        print(f"  {lab}: located {len(located[lab])}/{want[lab]} after trying {tried[lab]}")

    claims = [{**c, "source": "averitec", "label4": FOUR[c["gold_raw"]], "split": "test"} for c in v1]
    for lab, got in located.items():
        # interleave 5:3 (the TEST_ADD:DEV ratio) so a short pool still leaves a dev split to iterate on
        n_test = n_dev = 0
        for i, c in enumerate(got):
            to_test = n_dev >= DEV[lab] or (n_test < TEST_ADD[lab] and i % 8 < 5)
            claims.append({**c, "split": "test" if to_test else "dev"})
            n_test, n_dev = n_test + to_test, n_dev + (not to_test)
    write_jsonl(OUT / "claims.jsonl", claims)
    write_jsonl(OUT / "clips.jsonl", v1_clips + clips)
    test = [c for c in claims if c["split"] == "test"]
    stats = {
        "stopped_early_youtube_block": blocked >= 5, "test_claims": len(test), "dev_claims": len(claims) - len(test), "v1_claims": len(v1),
        "test_labels": Counter(c["label4"] for c in test), "dev_labels": Counter(c["label4"] for c in claims if c["split"] == "dev"),
        "test_sources": Counter(c["source"] for c in test), "test_videos": len({c["video"] for c in test}),
        "claimreview_candidates": len(cands), "tried": tried, "dropped": dropped,
        "fact_checkers": Counter(c.get("fact_checker", "AVeriTeC") for c in test),
        "located_by": Counter(c["located_by"] for c in test),
        "years": Counter((c.get("claim_date") or "?").split("-")[-1] for c in test),
    }
    stats = json.loads(json.dumps(stats, default=dict))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in ("test_claims", "dev_claims", "test_labels", "dev_labels", "test_videos")}, indent=1))


if __name__ == "__main__":
    main()
