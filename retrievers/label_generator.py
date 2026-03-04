"""Generate pseudo-labels using synonym replacement or random assignment."""

import random
from retrievers.pseudo_labels import ALL_LABELS

def generate_label(text: str = None):
    """
    Return a pseudo-label for input text.
    Here strictly randomized as per prompt to avoid LLM copy effects.
    text argument is kept for compatibility but not used for random assignment.
    """
    return random.choice(ALL_LABELS)

