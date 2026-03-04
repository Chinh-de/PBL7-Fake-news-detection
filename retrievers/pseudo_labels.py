"""
Collection of diverse pseudo-labels for semantic synonym replacement.
Categorized into Real (Authentic) and Fake (Hoax) to provide variety for the LLM.
"""

REAL_NEWS_LABELS = [
    # Basic
    "Real",
    "Authentic",
    "Reliable",
    "True",
    "Genuine",
    "Credible",
    "Fact-based",
    "Verified", 
    "Accurate",
    
    # Nuanced / Descriptive
    "Trustworthy",
    "Factual",
    "Legitimate",
    "Proven",
    "Substantiated",
    "Valid",
    "Confirmed",
    "Authoritative",
    "Honest",
    "Objective"
]

FAKE_NEWS_LABELS = [
    # Basic
    "Fake",
    "Hoax",
    "False",
    "Fabricated",
    "Untrue",
    "Misleading",
    "Bogus",
    "Inaccurate",
    
    # Nuanced / Descriptive
    "Dubious",
    "Deceptive",
    "Unverified",
    "Rumor",
    "Satire",
    "Propaganda",
    "Manipulated",
    "Distorted",
    "Baseless",
    "Phony",
    "Clickbait",
    "Disinformation",
    "Misinformation"
]

# Combined list for cases where we just need a random label pool without specific intent
ALL_LABELS = REAL_NEWS_LABELS + FAKE_NEWS_LABELS
