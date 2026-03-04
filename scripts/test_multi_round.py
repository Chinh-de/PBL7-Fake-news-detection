import sys
import os

# Add the source_code directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.slm_handler import DummySLM
from models.llm_handler import GeminiLLM
from core.multi_round_loop import run_multi_round_learning

def test_step_4_multi_round():
    print("--- Testing MRCD Step 4: Multi-Round Learning (Rounds 2-3) ---")
    
    # 1. Setup Initial Data (simulating end of Round 1)
    # D_clean: items that were confidently classified in Round 1
    d_clean = [
        {"text": "Clean Sample 1: The sun rises in the east.", "label": 0, "source": "Round1"}, # Real
        {"text": "Clean Sample 2: Water boils at 100 degrees Celsius.", "label": 0, "source": "Round1"}, # Real
        {"text": "Clean Sample 3: The earth is flat.", "label": 1, "source": "Round1"}, # Fake
        {"text": "Clean Sample 4: Humans have 3 legs.", "label": 1, "source": "Round1"}, # Fake
        {"text": "Clean Sample 5: Python is a programming language.", "label": 0, "source": "Round1"} # Real
    ]
    
    # D_noisy: items that were uncertain or disagreed upon in Round 1
    d_noisy = [
        {"text": "Noisy Sample 1: Coffee grants immortality.", "label": None}, # Likely Fake
        {"text": "Noisy Sample 2: Birds are government drones.", "label": None}, # Likely Fake
        {"text": "Noisy Sample 3: OpenAI released GPT-4.", "label": None}, # Likely Real
        {"text": "Noisy Sample 4: 1+1=2.", "label": None} # Likely Real
    ]
    
    # Mock Knowledge Cache (simulating retrieval from Round 1)
    knowledge_cache = {
        "Noisy Sample 1: Coffee grants immortality.": "Coffee contains caffeine which is a stimulant. There is no scientific evidence that it grants immortality.",
        "Noisy Sample 2: Birds are government drones.": "Birds are a group of warm-blooded vertebrates constituting the class Aves.",
        "Noisy Sample 3: OpenAI released GPT-4.": "GPT-4 is a multimodal large language model created by OpenAI.",
        "Noisy Sample 4: 1+1=2.": "Basic arithmetic states that one plus one equals two."
    }
    
    # 2. Initialize Models
    print("Initializing Models (GeminiLLM + DummySLM)...")
    llm = GeminiLLM()
    slm = DummySLM()
    
    # 3. Run Multi-Round Learning
    print(f"Initial State: D_clean={len(d_clean)}, D_noisy={len(d_noisy)}")
    final_clean, final_noisy = run_multi_round_learning(d_clean, d_noisy, knowledge_cache, slm, llm)
    
    # 4. Results
    print(f"\nFinal State: D_clean={len(final_clean)}, D_noisy={len(final_noisy)}")
    
    # New items in clean
    new_items_count = len(final_clean) - 5 # 5 was initial
    print(f"Items moved from Noisy to Clean: {new_items_count}")
    
    if new_items_count > 0:
        print("SUCCESS: Multi-round learning successfully moved items.")
        print("\nFirst 3 added items:")
        for item in final_clean[5:8]:
            print(f"- Text: {item['text'][:50]}... | Label: {item.get('label')} | Round: {item.get('round_added')}")
    else:
        print("WARNING: No items moved. This might be due to random confidence scores in DummySLM or LLM disagreement.")

if __name__ == "__main__":
    test_step_4_multi_round()
