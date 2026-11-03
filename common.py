"""Shared config, model registry, cached LLM client, JSON parsing and jsonl helpers."""
import asyncio
import hashlib
import html
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / ".cache"


def _load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

REGISTRY = json.loads((ROOT / "models.json").read_text())
MODELS = {m["key"]: m for m in REGISTRY["models"]}
SYSTEMS = REGISTRY["systems"]
PROVIDERS = {
    "fireworks": (os.environ.get("LLM_BASE_URL", "https://api.fireworks.ai/inference/v1"),
                  os.environ.get("LLM_API_KEY") or os.environ.get("FIREWORKS_API_KEY", "")),
    "openrouter": ("https://openrouter.ai/api/v1", os.environ.get("OPENROUTER_API_KEY", "")),
}


def resolve(model):
    """(key, provider, provider_model_id) for a registry key or a raw provider id."""
    m = MODELS.get(model)
    if m is None:
        return model, "fireworks", model
    mid = m["id"]
    if mid.startswith("env:"):
        mid = os.environ.get(mid[4:], "") or f"unset-{mid[4:]}"
    return m["key"], m["provider"], mid


MODEL = resolve(os.environ.get("MODEL", REGISTRY["base_open"]))[0]   # model under test
JUDGE_MODEL = resolve(os.environ.get("JUDGE_MODEL", REGISTRY["judge"]))[0]
MOCK = os.environ.get("LLM_BACKEND") == "mock"  # offline plumbing test, never for results
RESULTS = ROOT / ("results_mock" if MOCK else "results")

LABELS = ("supported", "contradicted", "unverified")
# AVeriTeC has four labels; the brief asks for three. Conflicting/cherrypicking -> unverified.
GOLD_MAP = {
    "Supported": "supported",
    "Refuted": "contradicted",
    "Not Enough Evidence": "unverified",
    "Conflicting Evidence/Cherrypicking": "unverified",
}

_clients = {}
_sem = None
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}


async def chat(messages, tag, model=None, temperature=0.0, max_tokens=2048, sample=0, judge=False, extra=None):
    """One chat completion, cached on disk by (model, messages, temperature, max_tokens, sample).

    `extra` goes to the provider as extra_body (e.g. {"reasoning": {"effort": "high"}} on OpenRouter).
    `tag` names the prompt type; the mock backend uses it to fake a reply. `sample` gives
    repeated draws of the same prompt (self-consistency voting) their own cache slots.
    Calls by the model under test (judge=False) count toward USAGE, cached or not, so cost
    and latency numbers are reproducible on reruns.
    """
    global _sem
    key_name, provider, mid = resolve(model or MODEL)
    key = hashlib.sha256(json.dumps([mid, messages, temperature, max_tokens] + ([sample] if sample else [])
                                    + ([extra] if extra else [])).encode()).hexdigest()
    f = CACHE / ("llm_mock" if MOCK else "llm") / f"{key}.json"
    if f.exists():
        out = json.loads(f.read_text())
    elif MOCK:
        out = {"text": _mock_reply(tag, messages, mid, sample), "prompt_tokens": 0, "completion_tokens": 0,
               "latency_s": 0.4 + (int(hashlib.md5((mid + tag).encode()).hexdigest(), 16) % 40) / 10}
    else:
        if provider not in _clients:
            from openai import AsyncOpenAI
            base, api_key = PROVIDERS[provider]
            _clients[provider] = AsyncOpenAI(base_url=base, api_key=api_key)
        if _sem is None:
            _sem = asyncio.Semaphore(int(os.environ.get("CONCURRENCY", 8)))
        async with _sem:
            for attempt in range(5):
                try:
                    t0 = time.perf_counter()
                    resp = await _clients[provider].chat.completions.create(
                        model=mid, messages=messages, temperature=temperature, max_tokens=max_tokens,
                        **({"extra_body": extra} if extra else {}))
                    latency = time.perf_counter() - t0
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(2 ** attempt)
        out = {"text": resp.choices[0].message.content or "",
               "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
               "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
               "latency_s": round(latency, 3)}
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out))
    if not judge:
        USAGE["calls"] += 1
        USAGE["prompt_tokens"] += out["prompt_tokens"]
        USAGE["completion_tokens"] += out["completion_tokens"]
        USAGE["latency_s"] += out.get("latency_s", 0.0)
    return out["text"]


def usage_cost(model=None):
    """USD for USAGE at the registry price of `model`; None when the price isn't set."""
    m = MODELS.get(resolve(model or MODEL)[0], {})
    p_in = os.environ.get("PRICE_IN") or m.get("price_in")
    p_out = os.environ.get("PRICE_OUT") or m.get("price_out")
    if p_in in (None, "") or p_out in (None, ""):
        return None
    return USAGE["prompt_tokens"] / 1e6 * float(p_in) + USAGE["completion_tokens"] / 1e6 * float(p_out)


def parse_json(text, default=None):
    """First JSON object in a model reply; tolerates <think> blocks and ```json fences."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {} if default is None else default


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", html.unescape(s or "").lower()))


def quote_in(quote, source):
    """True if the quote is (nearly) verbatim in the source: >=60% of its 4-word shingles appear."""
    q, s = norm(quote).split(), norm(source)
    if len(q) < 4:
        return bool(q) and " ".join(q) in s
    grams = [" ".join(q[i:i + 4]) for i in range(len(q) - 3)]
    return sum(g in s for g in grams) / len(grams) >= 0.6


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def field(text, name):
    """Value of a 'NAME: value' line in a prompt (used by the mock backend)."""
    m = re.search(rf"^{name}: (.*)$", text, flags=re.M)
    return m.group(1).strip() if m else ""


def _mock_reply(tag, messages, model_id="", sample=0):
    """Deterministic fake replies. Seeded by model, prompt and sample, so models differ at random."""
    user = messages[-1]["content"]
    h = int(hashlib.md5(f"{model_id}|{sample}|{user}".encode()).hexdigest(), 16)
    if tag == "extract":
        words = user.split("TRANSCRIPT:", 1)[-1].split()
        chunks = [" ".join(words[i:i + 18]) for i in range(0, min(len(words), 54), 18)]
        return json.dumps({"claims": [{"claim": c, "quote": c} for c in chunks if c]})
    if tag in ("queries", "questions"):
        return json.dumps({tag: [field(user, "CLAIM")[:100]]})
    if tag == "more":
        return json.dumps({"done": h % 3 != 0, "queries": [field(user, "CLAIM")[:60]]})
    if tag.startswith("verify") or tag == "repair":
        label = LABELS[h % 3]
        if tag == "verify_closed":
            cites = [{"url": "https://en.wikipedia.org/wiki/Fact-checking", "quote": "Fact-checking is the process"}]
        else:
            m = re.search(r"^\[1\] \S+ (.*)$", user, flags=re.M)
            cites = [{"id": 1, "quote": " ".join(m.group(1).split()[:10])}] if m else []
        return json.dumps({"label": label, "rationale": "mock", "citations": cites if label != "unverified" else []})
    if tag == "match":
        return json.dumps({"match": 1 if h % 2 else 0})
    if tag == "backs":
        return json.dumps({"backs": h % 2 == 0})
    if tag == "ground_truth":
        claim = field(user, "CLAIM")
        return json.dumps({"questions": [{"q": f"What exactly did the speaker say about {claim[:40]}?", "answer": "mock answer", "kind": "what"},
                                         {"q": "What do official figures show as of the video date?", "answer": "mock", "kind": "how_many"},
                                         {"q": "Is any context missing?", "answer": "mock", "kind": "context"}],
                           "subclaims": [{"text": claim[:80], "label": field(user, "FACT-CHECKER VERDICT").split(" ")[0] or "refuted"}],
                           "quantities": [{"stated": m, "actual": None, "unit": ""} for m in re.findall(r"\d[\d,.]*%?", claim)[:3]],
                           "traps": ["exact_number"] if re.search(r"\d", claim) else ["temporal"],
                           "difficulty": 1 + h % 5, "label_check": {"agrees": h % 9 != 0, "label": "conflicting", "note": "mock"}})
    if tag in ("v2_closed", "v2_label", "repair_v2"):
        labels = ("supported", "refuted", "conflicting", "not_enough_evidence")
        label = labels[h % 4]
        m = re.search(r"^\[1\] \S+ (.*)$", user, flags=re.M)
        quote = " ".join(m.group(1).split()[:10]) if m else ""
        cite = ([{"url": "https://en.wikipedia.org/wiki/Fact-checking", "quote": "Fact-checking is the process"}] if tag == "v2_closed"
                else [{"id": 1, "quote": quote}] if quote else [])
        decisive = label != "not_enough_evidence"
        return json.dumps({"label": label, "confidence": round(0.35 + (h % 60) / 100, 2), "claim_understood": field(user, "CLAIM")[:80],
                           "subclaims": [{"text": field(user, "CLAIM")[:60], "label": label}],
                           "questions": [{"q": "mock question one?", "answer": "mock", "citations": cite[:1]},
                                         {"q": "mock question two?", "answer": "no answer found", "citations": []}],
                           "rationale": "mock rationale for this claim based on the evidence provided here.",
                           "citations": cite if decisive else []})
    if tag == "triage":
        return json.dumps({"checkable": True, "needs_retrieval": h % 6 != 0, "kind": "number"})
    if tag == "decompose":
        return json.dumps({"subclaims": [{"text": field(user, "CLAIM")[:60], "questions": [field(user, "CLAIM")[:80]]}]})
    if tag == "critic":
        return json.dumps({"consistent": h % 4 != 0, "label": ("supported", "refuted", "conflicting", "not_enough_evidence")[h % 4], "reason": "mock"})
    if tag == "ideas":
        return json.dumps({"ideas": [{"name": f"Mock idea {i}", "type": "harness" if i % 2 else "post-training",
                                      "hypothesis": "mock", "mechanism": "mock", "expected_gain_pp": i, "added_seconds": i / 2,
                                      "build_hours": i, "steps": ["input", "mock step", "output"],
                                      "reference": "arXiv:2309.11495" if i % 2 else "arXiv:9999.99999"} for i in range(1, 6)]})
    if tag == "rubric":
        ids = re.findall(r"^- ([A-Z]\d+):", user, flags=re.M)
        return json.dumps({"scores": {i: (None if (h >> k) % 7 == 0 else (h >> k) % 2) for k, i in enumerate(ids)}})
    return "{}"
