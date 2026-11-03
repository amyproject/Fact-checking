"""Build the locked test set: short transcript clips + fact-checker-labeled claims.

Source: AVeriTeC (Schlichtkrull et al., NeurIPS 2023), train + dev. We keep claims whose
original post is a video, fetch the YouTube transcript, and cut long videos into a
~90 s clip around each claim so every test case looks like a short video.

Writes data/clips.jsonl, data/claims.jsonl, data/testset_stats.json.
"""
import json
import re
import urllib.request
from collections import Counter, defaultdict

import pandas as pd

from common import DATA, GOLD_MAP, write_jsonl
from retrieval import tokens

HF = "https://huggingface.co/datasets/pminervini/averitec/resolve/main/data/{}-00000-of-00001.parquet"
VIDEO_URL = re.compile(r"tiktok\.com|youtube\.com/(?:watch|shorts)|youtu\.be|instagram\.com/(?:reel|tv)/"
                       r"|facebook\.com/.*/videos|facebook\.com/watch|fb\.watch|rumble\.com|bitchute")
CLIP_SECONDS = 90       # target clip length
SHORT_VIDEO_MAX = 180   # videos at or under this are used whole
MIN_OVERLAP = 0.35      # share of a claim's content words that must appear in its clip


def load_averitec():
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ("train", "dev"):
        p = raw / f"averitec_{split}.parquet"
        if not p.exists():
            print(f"downloading AVeriTeC {split} ...")
            urllib.request.urlretrieve(HF.format(split), p)
        frames.append(pd.read_parquet(p).assign(split=split, row=lambda d: range(len(d))))
    return pd.concat(frames, ignore_index=True)


def video_key(url):
    # only YouTube URLs: Facebook's watch/?v=<number> would otherwise pass for a YouTube ID
    y = re.search(r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/)|youtu\.be/)([\w-]{11})", url)
    if y:
        return "yt:" + y.group(1)
    bare = url.split("web.archive.org/web/")[-1]
    return re.sub(r"\?.*", "", re.sub(r"^.*?https?://(www\.)?", "", bare))


def url_timestamp(url):
    """Seconds from a ?t=/&t= parameter (90, 90s, 1m30s), else None."""
    m = re.search(r"[?&#]t=(?:(\d+)h)?(?:(\d+)m)?(\d+)?s?", url)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s


def fetch_transcript(vid):
    """List of {start, end, text} in English, or (None, reason). Cached in data/transcripts/."""
    from youtube_transcript_api import YouTubeTranscriptApi
    f = DATA / "transcripts" / f"{vid}.json"
    if f.exists():
        cached = json.loads(f.read_text())
        return cached.get("snippets"), cached.get("reason")
    api = YouTubeTranscriptApi()
    snippets, reason = None, None
    try:
        tl = api.list(vid)
        try:
            tr = tl.find_transcript(["en", "en-US", "en-GB"])
        except Exception:
            tr = next(iter(tl))
            tr = tr.translate("en") if tr.is_translatable else None
        if tr is None:
            reason = "no English transcript"
        else:
            snippets = [{"start": s.start, "end": s.start + s.duration, "text": s.text.replace("\n", " ")}
                        for s in tr.fetch()]
    except Exception as e:
        reason = type(e).__name__
        # VideoUnplayable is also what YouTube returns while it is rate-limiting this IP
        if reason in ("IpBlocked", "RequestBlocked", "YouTubeRequestFailed", "VideoUnplayable"):
            return None, reason  # transient: don't cache, retry on the next run
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"snippets": snippets, "reason": reason}))
    return snippets, reason


def window_text(snips, start, end):
    return " ".join(s["text"] for s in snips if s["end"] > start and s["start"] < end)


def overlap(claim, text):
    c = set(tokens(claim))
    return len(c & set(tokens(text))) / max(len(c), 1)


def locate(claim, url, snips, duration):
    """(center_seconds, how, overlap) for where the claim is spoken."""
    if duration <= SHORT_VIDEO_MAX:
        return duration / 2, "whole video", overlap(claim, window_text(snips, 0, duration + 1))
    t = url_timestamp(url)
    if t is not None and t < duration:
        center = t + CLIP_SECONDS / 2
        return center, "url timestamp", overlap(claim, window_text(snips, t, t + CLIP_SECONDS))
    best = (0.0, 0.0)
    step = 15
    for start in range(0, max(int(duration - CLIP_SECONDS), 0) + step, step):
        ov = overlap(claim, window_text(snips, start, start + CLIP_SECONDS))
        if ov > best[0]:
            best = (ov, start)
    return best[1] + CLIP_SECONDS / 2, "lexical match", best[0]


def main():
    df = load_averitec()
    df = df[df.original_claim_url.fillna("").str.contains(VIDEO_URL)].copy()
    df["video"] = df.original_claim_url.map(video_key)
    stats = {"video_claims": len(df), "videos": df.video.nunique(), "dropped": Counter()}
    print(f"{len(df)} video-sourced claims across {df.video.nunique()} videos")

    claims, clips = [], []
    for vid, group in df.groupby("video"):
        if not vid.startswith("yt:"):
            youtube = "youtu" in vid
            stats["dropped"]["YouTube link without a video ID" if youtube else "non-YouTube (needs ASR)"] += len(group)
            continue
        snips, reason = fetch_transcript(vid[3:])
        if not snips:
            stats["dropped"][f"no transcript: {reason}"] += len(group)
            continue
        duration = snips[-1]["end"]
        located = []
        for _, r in group.iterrows():
            center, how, ov = locate(r.claim, r.original_claim_url, snips, duration)
            if ov < MIN_OVERLAP:
                stats["dropped"]["claim not found in transcript"] += 1
                continue
            located.append((center, how, ov, r))
        # claims spoken within one clip length of each other share a clip
        located.sort(key=lambda x: x[0])
        groups = []
        for item in located:
            if groups and item[0] - groups[-1][0][0] <= CLIP_SECONDS / 2:
                groups[-1].append(item)
            else:
                groups.append([item])
        for gi, grp in enumerate(groups):
            if duration <= SHORT_VIDEO_MAX:
                start, end = 0.0, duration
            else:
                start = max(0.0, grp[0][0] - CLIP_SECONDS / 2)
                end = min(duration, grp[-1][0] + CLIP_SECONDS / 2)
            clip_id = f"{vid[3:]}-{gi}"
            dates = [r.claim_date for _, _, _, r in grp if r.claim_date]
            clips.append({
                "clip_id": clip_id, "video": vid, "url": f"https://www.youtube.com/watch?v={vid[3:]}&t={int(start)}",
                "start": round(start, 1), "end": round(end, 1), "video_seconds": round(duration),
                "claim_date": min(dates, key=lambda d: d.split("-")[::-1]) if dates else None,
                "transcript": window_text(snips, start, end),
            })
            for center, how, ov, r in grp:
                claims.append({
                    "id": f"{r.split}-{r.row}", "clip_id": clip_id, "video": vid, "claim": r.claim,
                    "gold": GOLD_MAP[r.label], "gold_raw": r.label, "claim_date": r.claim_date,
                    "speaker": r.speaker, "claim_types": list(r.claim_types) if r.claim_types is not None else [],
                    "justification": r.justification, "fact_checking_article": r.fact_checking_article,
                    "original_claim_url": r.original_claim_url, "located_by": how, "overlap": round(ov, 2),
                })

    write_jsonl(DATA / "clips.jsonl", clips)
    write_jsonl(DATA / "claims.jsonl", claims)
    stats.update({
        "kept_claims": len(claims), "kept_clips": len(clips),
        "kept_videos": len({c["video"] for c in clips}),
        "labels": Counter(c["gold"] for c in claims), "labels_raw": Counter(c["gold_raw"] for c in claims),
        "located_by": Counter(c["located_by"] for c in claims),
        "claims_per_video": Counter(Counter(c["video"] for c in claims).values()),
        "clip_seconds_median": sorted(c["end"] - c["start"] for c in clips)[len(clips) // 2] if clips else None,
    })
    stats = json.loads(json.dumps(stats, default=dict))
    (DATA / "testset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
