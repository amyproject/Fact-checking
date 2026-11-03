"""The 50-item per-claim rubric and the trap taxonomy. Shared by 06_traps.py, 09_rubric.py and the report.

Every (claim, system) pair gets 50 checks in 5 areas. `auto` items are computed from the
scored row; `judge` items go to the judge model in one blind call per pair (it never sees
which model or system wrote the answer). An item can be N/A (a numeric check on a claim
with no number); N/A items leave the denominator. Claim score = sum over areas of
area weight x (passed / applicable) in that area, so 0-100.
"""

AREAS = {
    "A": {"name": "Verdict", "weight": 30, "desc": "Is the label right, and wrong in the least harmful way when it isn't?"},
    "B": {"name": "Evidence fidelity", "weight": 25, "desc": "Do the citations exist, say what is quoted, and back the label?"},
    "C": {"name": "Claim understanding", "weight": 15, "desc": "Did the system check the claim the speaker actually made?"},
    "D": {"name": "Reasoning", "weight": 15, "desc": "Does the rationale follow from the evidence without invented facts?"},
    "E": {"name": "Usefulness", "weight": 15, "desc": "Would a working fact-checker use this output?"},
}

# (id, method, applies-when, criterion). applies-when is a key resolved in 09_rubric.py.
ITEMS = [
    ("A1", "auto", "always", "Label matches the fact-checker's label"),
    ("A2", "auto", "always", "No supported <-> contradicted flip (the most harmful error)"),
    ("A3", "auto", "gold_decisive", "Does not abstain when the fact-checker reached a verdict"),
    ("A4", "auto", "gold_unverified", "Does not over-claim when the fact-checker found the evidence insufficient or cherry-picked"),
    ("A5", "judge", "always", "Verdict is judged as of the video date, not with later events"),
    ("A6", "judge", "trap:missing_context", "Flags the missing context that makes a technically true claim misleading"),
    ("A7", "judge", "always", "Verdict addresses the claim's checkable core, not a side detail"),
    ("A8", "judge", "trap:exact_number", "Compares the exact figure in the claim, not just its order of magnitude"),
    ("A9", "auto", "always", "End-to-end: the system extracted this claim from the transcript itself"),
    ("A10", "auto", "always", "End-to-end: its label on its own extracted wording matches the fact-checker"),

    ("B1", "auto", "pred_decisive", "Gives at least one citation for a supported/contradicted label"),
    ("B2", "auto", "has_citations", "Every cited source resolves"),
    ("B3", "auto", "has_citations", "Every quote is verbatim in its source (>=60% of 4-word shingles)"),
    ("B4", "auto", "pred_decisive", "At least one citation passes all three checks (resolves, verbatim, judge says it backs the label)"),
    ("B5", "auto", "always", "Cites no fact-checking site (they publish the answer key)"),
    ("B6", "auto", "always", "Does not cite this claim's own fact-check article or original post"),
    ("B7", "judge", "has_citations", "Cited sources are reliable for this claim (official data, primary documents, major outlets, encyclopedias)"),
    ("B8", "judge", "has_citations", "Cited sources were available on the video date (or are timeless)"),
    ("B9", "auto", "has_citations", "Citation precision: at least half of the citations pass all checks"),
    ("B10", "judge", "has_citations", "The cited sentence addresses the specific number, event or quote in the claim"),

    ("C1", "judge", "always", "Restates or treats the claim faithfully: no strawman, no softening"),
    ("C2", "judge", "always", "Uses who/when/where from the transcript where the claim depends on it"),
    ("C3", "judge", "trap:exact_number", "Keeps the speaker's numbers and units exactly"),
    ("C4", "judge", "has_traps", "Notices this claim's main trap (listed for the judge from the trap tags)"),
    ("C5", "judge", "matched", "The system's own extracted version of the claim stands alone (pronouns, place, time resolved)"),
    ("C6", "auto", "always", "Extraction is focused: 1-15 claims for this clip (not empty, not flooding)"),
    ("C7", "judge", "always", "Treats the claim as a factual statement, not an opinion or prediction (or explains why it can't be checked)"),
    ("C8", "judge", "trap:compound", "Covers every part of a compound claim, and labels by the weakest part"),
    ("C9", "judge", "always", "Copes with transcript noise without inventing what was said"),
    ("C10", "judge", "always", "Does not attribute words to the speaker that aren't in the transcript"),

    ("D1", "judge", "always", "Rationale is consistent with the label"),
    ("D2", "judge", "always", "Every fact in the rationale appears in the cited or retrieved evidence (no hallucinated facts)"),
    ("D3", "judge", "trap:causal", "Separates correlation from causation"),
    ("D4", "judge", "always", "Considers the plausible reading of ambiguous terms (which rate, which year, which population)"),
    ("D5", "judge", "trap:exact_number", "Arithmetic, percentages and unit conversions are correct"),
    ("D6", "judge", "always", "Matches the fact-checker's key reasoning (judge sees the fact-checker's justification)"),
    ("D7", "judge", "always", "Does not lean on the speaker's identity or party instead of evidence"),
    ("D8", "judge", "trap:temporal", "Temporal reasoning is right (\"last year\", \"since 2015\", \"record\" relative to the video date)"),
    ("D9", "auto", "always", "Rationale is 8-120 words"),
    ("D10", "judge", "pred_unverified", "For unverified: says what evidence would settle it"),

    ("E1", "judge", "always", "A fact-checker could publish this verdict with light edits"),
    ("E2", "judge", "always", "Rationale is concise and specific to this claim"),
    ("E3", "judge", "always", "Hedging matches the strength of the evidence"),
    ("E4", "auto", "always", "Output parsed as valid JSON with a valid label (no fallback)"),
    ("E5", "auto", "always", "Model calls per claim at most 3x the closed-book baseline's"),
    ("E6", "judge", "always", "Neutral, non-inflammatory tone"),
    ("E7", "judge", "always", "Flags when the 90 s excerpt is too little context to judge"),
    ("E8", "auto", "always", "Cites at most 3 sources (focused)"),
    ("E9", "judge", "always", "Saves a fact-checker time compared with starting from scratch"),
    ("E10", "judge", "always", "Makes no claims about people beyond what the evidence supports"),
]
RUBRIC = [{"id": i, "area": i[0], "method": m, "applies": a, "criterion": c} for i, m, a, c in ITEMS]
assert len(RUBRIC) == 50 and len({r["id"] for r in RUBRIC}) == 50

TRAPS = {
    "exact_number": {
        "name": "Exact number", "detect": "a digit, %, or a number word in the claim, or AVeriTeC tags it Numerical",
        "why": "Near-miss figures (7.3% vs 7.5%) and rounding decide the label; models accept a plausible-sounding number.",
        "failure": "Supports an approximately right figure, or rejects a correct figure stated in different units."},
    "missing_context": {
        "name": "Missing context", "detect": "label is Conflicting Evidence/Cherrypicking, or the justification says misleading / out of context / partly",
        "why": "The words are technically true but the impression is false; a binary true/false reading gets it wrong.",
        "failure": "Labels it supported because each word checks out."},
    "no_evidence": {
        "name": "No evidence exists", "detect": "fact-checker label is Not Enough Evidence",
        "why": "The right answer is to abstain; models prefer a confident verdict.",
        "failure": "Over-claims supported or contradicted from weak or unrelated evidence."},
    "local_knowledge": {
        "name": "Local knowledge", "detect": "claim location is outside the US and UK (AVeriTeC location code)",
        "why": "County budgets, local officials and national statistics are thin on Wikipedia and in pretraining.",
        "failure": "Retrieval returns other countries' facts; the model reasons from those."},
    "causal": {
        "name": "Causal claim", "detect": "AVeriTeC tags it Causal, or it says because / due to / caused / led to",
        "why": "Evidence usually shows co-occurrence, not cause.",
        "failure": "Supports a cause from a correlation."},
    "compound": {
        "name": "Compound claim", "detect": "two or more numbers, or several clauses joined by and / ; / ,",
        "why": "One true part can carry a false part.",
        "failure": "Labels the whole claim by its easiest part."},
    "attribution": {
        "name": "Attribution", "detect": "AVeriTeC tags Quote Verification or Position Statement, or the claim says said / claimed / according to",
        "why": "The question is whether someone said it, not whether it is true.",
        "failure": "Checks the content instead of the attribution."},
    "temporal": {
        "name": "Time-sensitive", "detect": "relative time (now, currently, last year, since, record, first time) or a year in the claim",
        "why": "The answer depends on the video date; models default to today.",
        "failure": "Uses later events or current figures."},
    "superlative": {
        "name": "Superlative", "detect": "most / largest / first / highest / lowest / never / always / only / every",
        "why": "Proving a superlative needs an exhaustive comparison that one passage rarely gives.",
        "failure": "Supports a superlative from one example."},
    "offline_source": {
        "name": "Needs an offline source", "detect": "fact-checker used Consultation (asked an official or expert) among its strategies",
        "why": "The deciding evidence was an interview or phone call, not a web page.",
        "failure": "No retrievable evidence; any verdict other than unverified is a guess."},
    "paraphrase_gap": {
        "name": "Paraphrase gap", "detect": "claim was located by lexical match with <50% word overlap",
        "why": "The fact-checker's wording differs from what was said, so extraction rarely produces a match.",
        "failure": "End-to-end miss: the system checks a different version of the claim."},
    "long_source": {
        "name": "Cut from a long video", "detect": "source video is longer than 10 minutes, so the clip is a 90 s window",
        "why": "Context that changes the meaning may sit outside the window.",
        "failure": "Misreads what the speaker referred to."},
}
