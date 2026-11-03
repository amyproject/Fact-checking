"""Rubric v2: gates, a cost matrix, calibration, QA scoring, robustness, efficiency, judge reliability.

Scoring one (claim, system) answer:
  1. 64 checks in 6 dimensions. `auto` checks are computed; `judge` checks go to two blind
     judges (primary on every answer, secondary on a 25% sample to measure agreement).
  2. Dimension score = weighted share of applicable checks passed (check weight x judge
     reliability: a judge check whose Cohen's kappa < 0.4 counts half).
  3. Raw score = sum over dimensions of dimension weight x dimension score (0-100).
  4. Gates: any failed gate caps the score at 25, however good the rest is.
  5. Headline metric, "verified accuracy": exact 4-way label AND every gate passes AND,
     for supported/refuted/conflicting, at least one citation passes all checks.
Separately reported: cost-weighted error (cost matrix below), Brier score, efficiency.
"""

LABELS4 = ("supported", "refuted", "conflicting", "not_enough_evidence")

DIMENSIONS = {
    "V": {"name": "Verdict", "weight": 25, "desc": "Exact 4-way label, harm-weighted errors, calibrated confidence"},
    "E": {"name": "Evidence", "weight": 20, "desc": "Real, verbatim, reliable, timely, independent sources"},
    "Q": {"name": "Verification QA", "weight": 15, "desc": "Asks the questions a fact-checker would, and answers them from evidence"},
    "R": {"name": "Reasoning", "weight": 15, "desc": "Decomposition, numbers, time, causality, consistency"},
    "B": {"name": "Robustness", "weight": 10, "desc": "Same answer under paraphrase and evidence reordering"},
    "U": {"name": "Efficiency & usefulness", "weight": 15, "desc": "Within budget, publishable, neutral"},
}

# Expected harm of answering `pred` when the truth is `gold` (0 = right, 1 = worst).
COST = {
    "supported":           {"supported": 0.0, "refuted": 1.0, "conflicting": 0.5, "not_enough_evidence": 0.3},
    "refuted":             {"supported": 1.0, "refuted": 0.0, "conflicting": 0.4, "not_enough_evidence": 0.3},
    "conflicting":         {"supported": 0.6, "refuted": 0.3, "conflicting": 0.0, "not_enough_evidence": 0.2},
    "not_enough_evidence": {"supported": 0.7, "refuted": 0.7, "conflicting": 0.4, "not_enough_evidence": 0.0},
}  # COST[gold][pred]

GATES = [
    ("G1", "No supported <-> refuted flip"),
    ("G2", "No fabricated citation: every quote is in its source and every URL exists"),
    ("G3", "No decisive evidence published after the video date"),
    ("G4", "Valid structured output (label, confidence, questions, citations parse)"),
    ("G5", "No statement about a person that the evidence does not support (judge)"),
]

# (id, dim, method, weight, applies-when, criterion)
ITEMS = [
    ("V1", "V", "auto", 3, "always", "Exact 4-way label matches the fact-checker"),
    ("V2", "V", "auto", 1, "always", "Collapsed 3-way label matches (supported / contradicted / unverified)"),
    ("V3", "V", "auto", 2, "always", "Harm-weighted error <= 0.3 (cost matrix)"),
    ("V4", "V", "auto", 2, "always", "Calibrated: Brier score of stated confidence <= 0.25"),
    ("V5", "V", "auto", 1, "gold_decisive", "Doesn't abstain when the fact-checker reached a verdict"),
    ("V6", "V", "auto", 1, "gold_nee", "Doesn't over-claim when evidence was insufficient"),
    ("V7", "V", "auto", 2, "gold_conflicting", "Recognizes a technically-true-but-misleading claim as conflicting"),
    ("V8", "V", "judge", 1, "always", "Verdict judged as of the video date"),
    ("V9", "V", "judge", 1, "always", "Verdict addresses the checkable core of the claim"),
    ("V10", "V", "auto", 1, "always", "End-to-end: the system extracted this claim from the transcript itself"),
    ("V11", "V", "auto", 1, "matched", "End-to-end: exact label on its own extracted wording"),

    ("E1", "E", "auto", 2, "pred_decisive", "At least one citation passes resolve + verbatim + backs-the-label"),
    ("E2", "E", "auto", 2, "has_citations", "Every quote is verbatim in its source"),
    ("E3", "E", "auto", 1, "has_citations", "Every cited source resolves"),
    ("E4", "E", "auto", 1, "always", "No fact-checking site or the claim's own fact-check cited"),
    ("E5", "E", "auto", 2, "has_citations", "Best source is tier 1-2 (official / intergovernmental / academic / major outlet)"),
    ("E6", "E", "auto", 1, "has_citations", "Mean source tier <= 2.5"),
    ("E7", "E", "auto", 1, "pred_decisive", "Two or more independent domains back the verdict"),
    ("E8", "E", "auto", 1, "has_gold_sources", "Evidence recall: shares a domain with the fact-checker's own sources"),
    ("E9", "E", "judge", 1, "has_citations", "Cited sources were available on the video date"),
    ("E10", "E", "judge", 2, "has_citations", "The cited sentence addresses the specific number, event or quote"),
    ("E11", "E", "auto", 1, "has_citations", "Citation precision >= 50%"),

    ("Q1", "Q", "auto", 1, "always", "Asks 2-5 verification questions"),
    ("Q2", "Q", "judge", 2, "has_gold_qa", "Its questions cover the fact-checker's key questions"),
    ("Q3", "Q", "judge", 2, "has_gold_qa", "Its answers agree with the fact-checker's answers"),
    ("Q4", "Q", "auto", 2, "has_questions", "Every non-empty answer cites evidence"),
    ("Q5", "Q", "auto", 1, "has_questions", "Answer quotes are verbatim in their passages"),
    ("Q6", "Q", "judge", 1, "has_questions", "Says \"no answer found\" instead of guessing when evidence is missing"),
    ("Q7", "Q", "judge", 1, "has_questions", "Questions are searchable (names, places, numbers, dates)"),
    ("Q8", "Q", "judge", 1, "has_questions", "The verdict follows from the answers"),
    ("Q9", "Q", "judge", 1, "trap:missing_context", "At least one question probes missing context"),
    ("Q10", "Q", "auto", 1, "has_questions", "No duplicate questions"),

    ("R1", "R", "judge", 2, "trap:compound", "Decomposes the claim into its sub-claims and labels each"),
    ("R2", "R", "auto", 2, "trap:compound", "Aggregation is consistent: any refuted sub-claim -> not supported"),
    ("R3", "R", "auto", 2, "trap:exact_number", "Numeric check: the claim's figure is compared with an evidence figure"),
    ("R4", "R", "judge", 1, "trap:exact_number", "Arithmetic, percentages and units are correct"),
    ("R5", "R", "judge", 1, "trap:temporal", "Temporal reasoning is right relative to the video date"),
    ("R6", "R", "judge", 1, "trap:causal", "Separates correlation from causation"),
    ("R7", "R", "judge", 2, "always", "Rationale is consistent with the label and the cited evidence"),
    ("R8", "R", "judge", 2, "always", "No fact in the rationale is absent from the evidence (no hallucination)"),
    ("R9", "R", "judge", 1, "always", "Matches the fact-checker's key reasoning"),
    ("R10", "R", "judge", 1, "always", "Resolves ambiguous terms to the plausible reading"),
    ("R11", "R", "judge", 1, "trap:attribution", "Checks attribution (who said it) separately from content"),

    ("B1", "B", "auto", 3, "has_paraphrase", "Same 4-way label when the claim is paraphrased"),
    ("B2", "B", "auto", 2, "has_shuffle", "Same 4-way label when evidence order is shuffled"),
    ("B3", "B", "auto", 1, "has_paraphrase", "Confidence moves < 0.2 under paraphrase"),
    ("B4", "B", "auto", 1, "has_votes", "Self-consistency: >= 4 of 5 samples agree"),

    ("U1", "U", "auto", 2, "always", "Model calls per claim within budget (<= 8)"),
    ("U2", "U", "auto", 2, "always", "Tokens per claim within budget (<= 12k)"),
    ("U3", "U", "auto", 1, "always", "Model seconds per claim within budget (<= 20 s)"),
    ("U4", "U", "judge", 2, "always", "A fact-checker could publish this with light edits"),
    ("U5", "U", "judge", 1, "always", "Concise and specific"),
    ("U6", "U", "judge", 1, "always", "Neutral tone"),
    ("U7", "U", "judge", 1, "always", "Flags when the excerpt is too little context"),
    ("U8", "U", "auto", 1, "always", "Rationale 15-150 words"),
    ("U9", "U", "auto", 1, "always", "At most 4 citations"),
    ("U10", "U", "judge", 1, "always", "Saves a fact-checker time versus starting from scratch"),
]
RUBRIC = [{"id": i, "dim": d, "method": m, "weight": w, "applies": a, "criterion": c} for i, d, m, w, a, c in ITEMS]
assert len({r["id"] for r in RUBRIC}) == len(RUBRIC)

# Source tiers for E5/E6: 1 official/intergovernmental/statistics, 2 academic or major outlet,
# 3 encyclopedia/reference, 4 other news, 5 blogs/social/unknown.
TIER1 = (".gov", ".gov.", ".go.ke", ".gov.ng", ".gov.za", ".gov.ph", ".gov.uk", ".mil", "who.int", "un.org", "worldbank.org",
         "imf.org", "oecd.org", "europa.eu", "census", "statistics", "bls.gov", "cdc.gov", "knbs.or.ke", "parliament", "congress.gov")
TIER2 = (".edu", ".ac.", "reuters.com", "apnews.com", "bbc.", "nytimes.com", "washingtonpost.com", "theguardian.com",
         "nature.com", "science.org", "nih.gov", "jstor.org", "ft.com", "wsj.com", "economist.com", "npr.org", "pewresearch.org")
TIER3 = ("wikipedia.org", "britannica.com", "wikidata.org")
TIER5 = ("facebook.com", "twitter.com", "x.com", "youtube.com", "tiktok.com", "instagram.com", "reddit.com", "blogspot.",
         "wordpress.", "medium.com", "substack.com")


def source_tier(url):
    u = (url or "").lower()
    for tier, keys in ((1, TIER1), (2, TIER2), (3, TIER3), (5, TIER5)):
        if any(k in u for k in keys):
            return tier
    return 4 if u.startswith("http") else 5


def applies(rule, a):
    """`a` is the answer context built by 14_score_v2.py."""
    if rule == "always":
        return True
    if rule.startswith("trap:"):
        return rule[5:] in a["traps"]
    return bool(a.get(rule))


def kappa(pairs):
    """Cohen's kappa for binary judgments [(primary, secondary), ...]; None if too few."""
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 8:
        return None
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    p1, p2 = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def claim_score(items, gates, kappas=None):
    """0-100 score for one answer: weighted dimension shares, judge checks down-weighted when
    their kappa < 0.4, capped at 25 if any gate fails. Returns (score, {dim: share})."""
    kappas = kappas or {}
    dims = {}
    for it in RUBRIC:
        v = items.get(it["id"])
        if v is None:
            continue
        w = it["weight"] * (0.5 if it["method"] == "judge" and (kappas.get(it["id"]) or 1) < 0.4 else 1)
        num, den = dims.get(it["dim"], (0.0, 0.0))
        dims[it["dim"]] = (num + w * v, den + w)
    shares = {d: (n / w if w else None) for d, (n, w) in dims.items()}
    live = {d: s for d, s in shares.items() if s is not None}
    raw = 100 * sum(DIMENSIONS[d]["weight"] * s for d, s in live.items()) / max(sum(DIMENSIONS[d]["weight"] for d in live), 1)
    return (min(raw, 25.0) if not all(gates.values()) else raw), shares
