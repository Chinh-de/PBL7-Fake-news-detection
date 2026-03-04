import requests
import csv
import json
import os
import io

# Using a raw raw github url for AG News dataset
# This is a common location for the dataset in CSV format
# Format: "Class Index", "Title", "Description"
URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "news_corpus.json")

def fetch_and_save_news():
    print(f"Attempting to download AG News dataset from {URL}...")
    try:
        response = requests.get(URL)
        response.raise_for_status()
        
        print("Download successful. Processing...")
        
        # Use csv module to parse the content
        # The content is text, so we wrap it in StringIO
        f = io.StringIO(response.text)
        reader = csv.reader(f, delimiter=',', quotechar='"')
        
        news_entries = []
        limit = 5000 
        
        for i, row in enumerate(reader):
            if i >= limit:
                break
            
            # Row structure: [Class Index, Title, Description]
            if len(row) >= 3:
                title = row[1]
                description = row[2]
                # Combine matching typical article structure
                content = f"{title}\n{description}"
                
                news_entries.append({
                    "id": str(i),
                    "content": content,
                    "source": "ag_news"
                })
        
        print(f"Extracted {len(news_entries)} articles.")
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(news_entries, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Failed to fetch or process dataset: {e}")
        # Fallback to creating a small dummy corpus if download fails, 
        # so the system has *something* to work with.
        create_dummy_corpus()

def create_dummy_corpus():
    print("Creating specific dummy news corpus as fallback...")
    dummy_data = [
        {"id": "0", "content": "Technology stocks rose today as the market reacted to new AI developments.", "source": "dummy"},
        {"id": "1", "content": "Thelocal sports team won the championship in a stunning upset last night.", "source": "dummy"},
        {"id": "2", "content": "New environmental policies were announced by the government to combat climate change.", "source": "dummy"},
        {"id": "3", "content": "Health experts advise eating more fruits and vegetables for a balanced diet.", "source": "dummy"},
        {"id": "4", "content": "SpaceX successfully launched another batch of satellites into orbit.", "source": "dummy"}
    ]
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, indent=2, ensure_ascii=False)
    print(f"Created dummy corpus with {len(dummy_data)} items at {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_and_save_news()
