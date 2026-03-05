import sys
import os
import json
import re

# Add parent dir to sys.path to allow importing from sibling directories when running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm_handler import call_llm

try:
    import wikipedia
except ImportError:
    wikipedia = None
    print("Warning: wikipedia module not found. Install with `pip install wikipedia`")

"""LLM agent for extracting entities and querying Wikipedia."""


def extract_entities(text: str) -> list[dict]:
    """
    Extract key entities from text using the configured LLM, along with their likely Wikipedia language.
    Returns a list of dicts: [{'entity': 'Name', 'lang': 'vi'}, ...]
    """
   
    prompt = (
    f"You are a knowledge extraction agent for a fact-checking system. "
    f"Identify the top 1 to 4 most important named entities (People, Organizations, Locations, Events) "
    f"in the text below that are crucial for verifying the event and are likely to have a Wikipedia page. "
    f"For each entity, determine the most appropriate Wikipedia language code (e.g., 'en', 'vi'). "
    f"Return ONLY a valid, raw JSON array of objects, where each object has exactly two keys: 'entity' and 'lang'. "
    f"Absolutely NO markdown formatting (do not use ```json), NO introductions, and NO explanations.\n\n"
    f"Example output:\n"
    f"[{{\"entity\": \"Germanwings\", \"lang\": \"en\"}}, {{\"entity\": \"French Alps\", \"lang\": \"en\"}}]\n\n"
    f"Text: \"{text}\"\n\n"
    f"JSON:"
    )
    
    try:
        response = call_llm(prompt)
        # Attempt to clean potential markdown wrappers
        clean_response = response.replace("```json", "").replace("```", "").strip()
        
        # Use regex to find the JSON array part if there's extra text
        match = re.search(r'\[.*\]', clean_response, re.DOTALL)
        if match:
            clean_response = match.group(0)
            
        entities = json.loads(clean_response)
        
        # Ensure it's a list of dicts
        if isinstance(entities, list):
            return entities
        return []
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []


def query_wikipedia(entity: str, lang: str = "en") -> str:
    """Query Wikipedia for a summary of the entity."""
    if wikipedia is None:
        return "Wikipedia module not available."
        
    try:
        wikipedia.set_lang(lang)
        summary = wikipedia.summary(entity)
        return summary
    except Exception:
        return "Not found"


def extract_and_summarize(text: str) -> dict[str, str]:
    entities_data = extract_entities(text)

    raw_results = []   # [(entity, summary)]
    
    if not entities_data:
        return {}

    # ===== Phase 1: Query Wikipedia =====
    for item in entities_data:
        if not isinstance(item, dict):
            continue

        entity = item.get('entity')
        lang = item.get('lang', 'en')

        if not entity:
            continue

        summary = query_wikipedia(entity, lang=lang)

        # bỏ qua not found
        if not summary or summary == "Not found" or "Error" in summary:
            continue

        raw_results.append((entity, summary.strip()))

    # ===== Phase 2: Merge by identical summary =====
    summary_map = {}   # summary_text -> [entity1, entity2, ...]

    for entity, summary in raw_results:
        if summary in summary_map:
            if entity not in summary_map[summary]:
                summary_map[summary].append(entity)
        else:
            summary_map[summary] = [entity]

    # Build final result
    merged_results = {}
    for summary_text, entities in summary_map.items():
        merged_key = ", ".join(entities)
        merged_results[merged_key] = summary_text

    return merged_results
