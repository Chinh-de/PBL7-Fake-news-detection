import sys
import os
import json

# Add the source_code directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers.bing_news_seach import search_news
from retrievers.bm25_matcher import retrieve_demonstrations, load_news_corpus 
from retrievers.wiki_agent import extract_and_summarize
from models.llm_handler import set_llm, GeminiLLM, MockLLM

def test_demonstration_retrieval():
    # Setup LLM for extraction test
    if os.environ.get("GEMINI_API_KEY"):
        print("LLM: Gemini (Found API Key)")
        set_llm(GeminiLLM())
    else:
        print("LLM: Mock (No API Key found)")
        set_llm(MockLLM())

    query = "AI dominated programmers"
    print(f"Testing retrieval for query: '{query}'")
    
    # 1. Get N_w from Bing
    print("Fetching news from Bing...")
    # Updated call: search_news returns list of strings now, and uses max_results
    bing_news_strings = search_news(query, max_results=10)
    print(f"Found {len(bing_news_strings)} articles from Bing.")
    
    # search_news returns strings, but retrieve_demonstrations expects dicts with "content"
    # We wrap them here to match the expected format for BM25 processing
    bing_news = [{"content": text, "source": "bing_search"} for text in bing_news_strings]
    
    # 2. Retrieve demonstrations (includes loading N_c via internal call or external call if needed)
    # The retrieve_demonstrations function in bm25_matcher handles merging N_w + N_c
    print("Retrieving top 4 demonstrations with pseudo-labels...")
    demonstrations = retrieve_demonstrations(query, bing_news, k=4)
    
    print("\n--- Results ---\n")
    print(json.dumps(demonstrations, indent=2, ensure_ascii=False))
    
    if len(demonstrations) == 4:
        print("\nSUCCESS: Retrieved exactly 4 demonstrations as requested.")
    else:
        print(f"\nWARNING: Retrieved {len(demonstrations)} demonstrations instead of 4.")

if __name__ == "__main__":
    test_demonstration_retrieval()

    # 3. Knowledge Retrieval (Wiki Agent)
    print("\n--- Testing Knowledge Retrieval (Wiki Agent) ---")
    query = "Donald Trump elected as president in 2016"
    knowledge_dict = extract_and_summarize(query)
    print("Knowledge Extraction Results:")
    print(json.dumps(knowledge_dict, indent=2, ensure_ascii=False))
