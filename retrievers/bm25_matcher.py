"""BM25 retrieval using rank_bm25 library."""

import json
import os
import random
from rank_bm25 import BM25Okapi
from retrievers.label_generator import generate_label

# Path to the static news corpus
NEWS_CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "news_corpus.json")

def load_news_corpus():
    """Load the static news corpus from JSON file."""
    if not os.path.exists(NEWS_CORPUS_PATH):
        print(f"Warning: News corpus not found at {NEWS_CORPUS_PATH}")
        return []
    try:
        with open(NEWS_CORPUS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading news corpus: {e}")
        return []

def retrieve_demonstrations(query: str, bing_news_items: list, k: int = 4):
    """
    Merge Bing news with static corpus, rank by BM25, and select top k.
    Apply pseudo-labeling to the selected K items.
    
    Args:
        query (str): The input query/claim to verify.
        bing_news_items (list): List of news dicts from Bing search.
        k (int): Number of demonstrations to return.
        
    Returns:
        list: List of dicts with 'text' and 'label' keys.
    """
    # 1. Merge N_w (Bing) and N_c (Corpus) -> N_total
    static_corpus = load_news_corpus()
    total_corpus = bing_news_items + static_corpus
    
    if not total_corpus:
        return []

    # 2. Prepare corpus for BM25
    # BM25 requires tokenized corpus. Simple split() is used here as per original simple implementation.
    # For better results, a proper tokenizer should be used, but this follows the established pattern.
    tokenized_corpus = [doc.get("content", "").lower().split() for doc in total_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 3. Get scores for the query
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # 4. Get top K indices
    # We zip scores with indices to sort
    scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, score in scored_indices[:k]]
    
    # 5. Extract top K documents
    top_docs = [total_corpus[i] for i in top_k_indices]
    
    # 6. Apply Pseudo-Labeling
    demonstrations = []
    for doc in top_docs:
        text = doc.get("content", "")
        # Randomly assign a label from the synonym set
        label = generate_label(text)
        
        demonstrations.append({
            "text": text,
            "label": label,
            "source": doc.get("source", "unknown")
        })
        
    return demonstrations

def retrieve_demonstrations_from_clean(query: str, clean_data: list, k: int = 4):
    """
    Retrieve top-k demonstrations from D_clean using BM25.
    Unlike retrieve_demonstrations, this uses existing labels from clean_data.
    
    Args:
        query (str): The input text to find similar examples for.
        clean_data (list): List of dicts (the D_clean dataset), each having 'text' and 'label'.
        k (int): Number of demonstrations.
        
    Returns:
        list: List of dicts with 'text', 'label' (0 or 1), 'source'.
    """
    if not clean_data:
        return []
        
    # Prepare corpus from clean data
    # Tokenizing text for BM25
    tokenized_corpus = [doc.get("text", "").lower().split() for doc in clean_data]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Query scoring
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get Top K indices
    # Sorting by score descending
    scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, score in scored_indices[:k]]
    
    top_docs = [clean_data[i] for i in top_k_indices]
    
    # Format output (labels are already present, no pseudo-labeling)
    formatted_docs = []
    for doc in top_docs:
        formatted_docs.append({
            "text": doc.get("text", ""),
            "label": doc.get("label", 0), # Default to 0 if missing, but should be there
            "source": "D_clean_iterative"
        })
        
    return formatted_docs

def match(corpus, query):
    """Return top documents matching query."""
    # ... existing implementation kept for compatibility if needed ...
    bm25 = BM25Okapi([doc.split() for doc in corpus])
    scores = bm25.get_scores(query.split())
    return scores
