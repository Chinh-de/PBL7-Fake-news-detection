from ddgs import DDGS
import json

def search_news(query: str, max_results: int = 10):
    """
    Search for news using DuckDuckGo with bing news engine and return concatenated string of title + body.
    
    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return.
        
    Returns:
        str: concatenated news string
    """
    
    news_items = []
    
    try:
        with DDGS() as ddgs:
            results_gen = ddgs.news(
                query=query,
                region="wt-wt",
                safesearch="off",
                timelimit=None, 
                max_results=max_results,
                backend="bing"
            )
            
            for i, result in enumerate(results_gen):
                if i >= max_results:
                    break
                
                title = result.get("title", "")
                body = result.get("body", "")
                content = f"{title}\n{body}"
                
                news_items.append(content)
                
    except Exception as e:
        print(f"Error searching DuckDuckGo: {e}")
        
    return news_items

# Test
if __name__ == "__main__":
    test_query = "AI dominated programmers"
    news_list = search_news(test_query, max_results=10)
    print(f"Retrieved {len(news_list)} news items:")
    for idx, news in enumerate(news_list):
        print(f"\n--- News Item {idx+1} ---\n{news}")