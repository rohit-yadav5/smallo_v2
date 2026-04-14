"""memory_system/entities/extractor.py — Named-entity extraction using spaCy.

Replaces the previous LLM-based extractor (ask_llm() call, 5-8 s latency).
spaCy en_core_web_sm runs in ~20 ms with no cold-load penalty after first import.
Model is lazy-loaded via lru_cache so the first call pays the load cost (~150 ms)
and every subsequent call is instant.

Output format is backward-compatible with insert_pipeline.py and retrieval/search.py:
  {"name": str, "domain": str, "category": str, "entity_type": str}
"""

import re
import logging
from functools import lru_cache

from memory_system.entities.registry import CONTROLLED_ENTITIES

_log = logging.getLogger(__name__)

# ── spaCy label → internal entity_type mapping ────────────────────────────────
# Only keep entity types that are useful for memory retrieval; discard noisy
# labels like CARDINAL, ORDINAL, QUANTITY, PERCENT, MONEY.
_SPACY_TO_ENTITY_TYPE: dict[str, str] = {
    "PERSON":       "Concept",          # people → Concept (no "person" type in schema)
    "ORG":          "Service",          # organisations
    "GPE":          "Concept",          # countries/cities/states
    "LOC":          "Concept",          # non-GPE locations
    "PRODUCT":      "Tool",             # product names
    "EVENT":        "Concept",          # named events
    "WORK_OF_ART":  "Concept",
    "LAW":          "Concept",
    "LANGUAGE":     "Technology",
    "FAC":          "Infrastructure",   # facilities / buildings
    "NORP":         "Concept",          # nationalities / groups
}

# Domain assigned by spaCy entity label
_SPACY_TO_DOMAIN: dict[str, str] = {
    "PERSON":       "Personal",
    "ORG":          "Engineering",
    "GPE":          "External",
    "LOC":          "External",
    "PRODUCT":      "Engineering",
    "EVENT":        "External",
    "WORK_OF_ART":  "Learning",
    "LAW":          "System",
    "LANGUAGE":     "Engineering",
    "FAC":          "External",
    "NORP":         "External",
}

# Category assigned by spaCy entity label
_SPACY_TO_CATEGORY: dict[str, str] = {
    "PERSON":       "Concept",
    "ORG":          "Service",
    "GPE":          "Concept",
    "LOC":          "Concept",
    "PRODUCT":      "Tool",
    "EVENT":        "Concept",
    "WORK_OF_ART":  "Concept",
    "LAW":          "Concept",
    "LANGUAGE":     "Technology",
    "FAC":          "Infrastructure",
    "NORP":         "Concept",
}

# Known tech/infra keywords — fast O(n) scan, no model needed.
# These supplement spaCy for tokens the small model often misclassifies.
TECH_KEYWORDS: set[str] = {
    "redis", "postgresql", "postgres", "mysql", "sqlite",
    "docker", "kubernetes", "faiss", "chromadb",
    "fastapi", "flask", "django", "react", "vue", "angular",
    "kafka", "rabbitmq", "celery",
    "python", "javascript", "typescript", "golang", "rust",
    "aws", "gcp", "azure", "terraform", "ansible",
    "nginx", "gunicorn", "uvicorn",
    "ollama", "whisper", "kokoro",
}

# Short conversational inputs — skip processing entirely
_CONVERSATIONAL_SKIP = re.compile(
    r"^(hello|hi|hey|ok|okay|yes|no|thanks|thank you|sure|great|"
    r"how are you|what's up|wassup|good|bye|goodbye|stop|quit|exit|"
    r"hmm|uh|uhh|oh|ah|alright|cool|nice|got it|sounds good)[\s!?.]*$",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _get_nlp():
    """Load spaCy model once, cached forever (lazy init)."""
    import spacy  # noqa: PLC0415 — intentional deferred import
    return spacy.load("en_core_web_sm")


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities from text using spaCy + controlled registry + tech keywords.

    Returns list of dicts with keys:
        name        str  — lowercase canonical entity name
        domain      str  — Engineering|Learning|Personal|System|External
        category    str  — Technology|Service|Concept|Tool|Infrastructure|…
        entity_type str  — matches ALLOWED_ENTITY_TYPES from old extractor

    Never raises — returns [] on any error.
    """
    if not text or not text.strip():
        return []

    # Skip trivial conversational inputs
    if _CONVERSATIONAL_SKIP.match(text.strip()):
        return []

    # Skip very short inputs (< 4 words) — not enough signal for NER
    if len(text.split()) < 4:
        return []

    text_lower = text.lower()
    extracted: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, domain: str, category: str, entity_type: str) -> None:
        """Dedup-safe add."""
        key = name.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        extracted.append({
            "name":        key,
            "domain":      domain,
            "category":    category,
            "entity_type": entity_type,
        })

    # ── 1. Controlled registry (highest priority) ──────────────────────────
    for group in CONTROLLED_ENTITIES.values():
        for name, meta in group.items():
            if name in text_lower:
                _add(name, meta["domain"], meta["category"], meta["entity_type"])

    # ── 2. Direct tech keyword scan (O(n), no model) ──────────────────────
    for keyword in TECH_KEYWORDS:
        if keyword in text_lower:
            _add(keyword, "Engineering", "Technology", "Technology")

    # ── 3. Rule-based: Python filenames and port numbers ──────────────────
    for fname in re.findall(r"\b\w+\.py\b", text):
        _add(fname, "Engineering", "Infrastructure", "File")

    for port in re.findall(r":(\d{2,5})\b", text):
        _add(f"port_{port}", "Engineering", "Infrastructure", "Port")

    # ── 4. spaCy NER (replaces LLM call — ~20 ms, no cold-load) ──────────
    try:
        nlp = _get_nlp()
        doc = nlp(text)
        for ent in doc.ents:
            label = ent.label_
            etype = _SPACY_TO_ENTITY_TYPE.get(label)
            if etype is None:
                continue  # skip CARDINAL, MONEY, PERCENT, etc.
            domain   = _SPACY_TO_DOMAIN.get(label, "External")
            category = _SPACY_TO_CATEGORY.get(label, "Concept")
            _add(ent.text, domain, category, etype)
    except Exception as exc:
        _log.warning("spaCy NER failed (non-fatal): %s", exc)

    return extracted


def extract_entity_names(text: str) -> list[str]:
    """Convenience wrapper — returns just entity name strings."""
    return [e["name"] for e in extract_entities(text)]
