import re
import json
from memory_system.entities.registry import CONTROLLED_ENTITIES
from llm import ask_llm


# -----------------------------
# Allowed Hierarchy Constraints
# -----------------------------

ALLOWED_DOMAINS = {
    "Engineering",
    "Learning",
    "Personal",
    "System",
    "External"
}

ALLOWED_CATEGORIES = {
    "Project",
    "Technology",
    "Infrastructure",
    "Concept",
    "Tool",
    "Framework",
    "Database",
    "Service",
    "Component",
    "Feature"
}

ALLOWED_ENTITY_TYPES = {
    "Technology",
    "Database",
    "Infrastructure",
    "Concept",
    "Tool",
    "Framework",
    "Service",
    "Component",
    "Feature",
    "Parent",
    "File",
    "Port"
}

COMMON_ACTION_WORDS = {
    "integrated",
    "configured",
    "implemented",
    "optimized",
    "improved",
    "fixed",
    "added",
    "built",
    "created",
    "designed",
    "developed"
}

TECH_KEYWORDS = {
    "redis",
    "postgresql",
    "postgres",
    "docker",
    "faiss",
    "chromadb",
    "fastapi",
    "kafka",
    "mysql"
}


# -----------------------------
# Normalization Helpers
# -----------------------------

def normalize_domain(domain_value: str):
    if not domain_value:
        return "Engineering"

    parts = [d.strip() for d in domain_value.split("/")]

    for part in parts:
        if part in ALLOWED_DOMAINS:
            return part

    return "Engineering"


def normalize_category(category_value: str):
    if not category_value:
        return "Technology"

    for allowed in ALLOWED_CATEGORIES:
        if allowed.lower() in category_value.lower():
            return allowed

    return "Technology"


def normalize_entity_type(entity_type_value: str):
    """
    Normalize entity_type into controlled set.
    """
    if not entity_type_value:
        return "Concept"

    for allowed in ALLOWED_ENTITY_TYPES:
        if allowed.lower() in entity_type_value.lower():
            return allowed

    return "Concept"


def refine_entity_name(name: str):
    """
    Hybrid name refinement:
    1. Strip action verbs
    2. Detect known tech keywords
    3. Reduce to core noun tokens
    4. Fallback to LLM refinement
    """

    if not name:
        return None

    name = name.strip().lower()

    # Remove leading action words
    tokens = name.split()
    tokens = [t for t in tokens if t not in COMMON_ACTION_WORDS]

    if not tokens:
        return None

    # If contains known tech keyword, reduce to that
    for token in tokens:
        if token in TECH_KEYWORDS:
            return token

    # If phrase too long, reduce to last 2 words
    if len(tokens) > 3:
        tokens = tokens[-2:]

    refined = " ".join(tokens)

    # If still too long or messy → LLM refinement
    if len(refined) > 40:
        try:
            prompt = f"""
Reduce the following phrase to its core technical entity name.
Only return the entity name.

Phrase: {refined}
"""
            response = ask_llm(prompt)
            return response.strip().lower()
        except Exception:
            return refined

    return refined


def safe_parse_json(response: str):
    try:
        start = response.find("[")
        end = response.rfind("]")

        if start == -1 or end == -1:
            return []

        json_str = response[start:end + 1]
        return json.loads(json_str)

    except Exception:
        return []


# -----------------------------
# Main Extraction Function
# -----------------------------

_CONVERSATIONAL_SKIP = re.compile(
    r"^(hello|hi|hey|ok|okay|yes|no|thanks|thank you|sure|great|"
    r"how are you|what's up|wassup|good|bye|goodbye|stop|quit|exit|"
    r"hmm|uh|uhh|oh|ah|alright|cool|nice|got it|sounds good)[\s!?.]*$",
    re.IGNORECASE
)


def extract_entities(text: str):

    text_lower = text.lower()
    extracted = []

    # Skip LLM entirely for short or purely conversational input
    words = text.split()
    if len(words) < 6 or _CONVERSATIONAL_SKIP.match(text.strip()):
        return extracted

    # 1️⃣ Controlled Registry
    for group in CONTROLLED_ENTITIES.values():
        for name, meta in group.items():
            if name in text_lower:
                extracted.append({
                    "name": name,
                    "domain": meta["domain"],
                    "category": meta["category"],
                    "entity_type": meta["entity_type"]
                })

    # 1.5️⃣ Direct Tech Keyword Detection (LLM-independent fallback)
    for keyword in TECH_KEYWORDS:
        if keyword in text_lower:
            extracted.append({
                "name": keyword,
                "domain": "Engineering",
                "category": "Technology",
                "entity_type": "Technology"
            })

    # 2️⃣ Rule-Based (files + ports)
    files = re.findall(r"\b\w+\.py\b", text)
    for f in files:
        extracted.append({
            "name": f,
            "domain": "Engineering",
            "category": "Infrastructure",
            "entity_type": "File"
        })

    ports = re.findall(r":(\d{2,5})", text)
    for port in ports:
        extracted.append({
            "name": f"port_{port}",
            "domain": "Engineering",
            "category": "Infrastructure",
            "entity_type": "Port"
        })

    # 3️⃣ LLM Extraction
    try:
        prompt = f"""
Extract structured entities from the following text.

Text:
{text}

Return JSON list:
[
  {{
    "name": "...",
    "domain": "Engineering|Learning|Personal|System|External",
    "category": "...",
    "entity_type": "..."
  }}
]

Only return JSON.
"""

        response = ask_llm(prompt)
        llm_entities = safe_parse_json(response)

        for entity in llm_entities:

            if not all(k in entity for k in ("name", "domain", "category", "entity_type")):
                continue

            refined_name = refine_entity_name(entity["name"])
            if not refined_name:
                continue

            domain = normalize_domain(entity["domain"])
            category = normalize_category(entity["category"])

            entity_type = normalize_entity_type(entity["entity_type"])

            extracted.append({
                "name": refined_name,
                "domain": domain,
                "category": category,
                "entity_type": entity_type
            })

    except Exception:
        pass

    # Deduplicate
    unique = {}
    for entity in extracted:
        key = entity["name"].strip().lower()
        if key not in unique:
            unique[key] = entity

    return list(unique.values())