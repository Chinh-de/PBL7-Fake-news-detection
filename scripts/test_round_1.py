import sys
import os

# Add the source_code directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.slm_handler import DummySLM
from models.llm_handler import GeminiLLM
from core.selection_module import run_selection_pipeline

def test_round_1_logic():
    print("--- Testing MRCD Step 3: Round 1 (Zero-shot & Classification) ---")
    
    # 1. Inputs
    text_x = "Scientists confirm new species of flying penguin discovered in Antarctica."
    
    # Mock Demonstrations (k=4)
    demonstrations_D = [
        {"text": "Breaking: Aliens land in Times Square.", "label": "Fake"},
        {"text": "NASA launches new rover to Mars.", "label": "Real"},
        {"text": "Study shows chocolate cures all diseases.", "label": "Fake"},
        {"text": "Government passes new tax bill.", "label": "Real"}
    ]
    
    # Mock Knowledge
    knowledge_K = "Penguins are flightless birds found in the Southern Hemisphere. No flying species exists."
    
    # 2. Initialize Models
    # Using real GeminiLLM as requested to test full pipeline integration
    print("Initializing GeminiLLM (ensure GEMINI_API_KEY is set in .env)...")
    llm = GeminiLLM()
    slm = DummySLM()

    # 3. Run Pipeline
    result = run_selection_pipeline(text_x, demonstrations_D, knowledge_K, llm=llm, slm=slm)
    
    # 4. Assertions / Verification
    print("\n--- Final Result ---")
    print(f"Input Text: {result['text']}")
    print(f"LLM Label: {result['y_hat_1']}")
    print(f"SLM Label: {result['y_hat_2']} (Conf: {result['confidence']:.2f})")
    print(f"Assigned Category: {result['category']}")
    
    # Verify logic manually
    expected_category = "clean" if (result['y_hat_1'] == result['y_hat_2'] and result['confidence'] >= 0.8) else "noisy"
    if result['category'] == expected_category:
        print("SUCCESS: Category assignment logic is correct.")
    else:
        print(f"FAILURE: Logic mismatch! Expected {expected_category}, got {result['category']}")

if __name__ == "__main__":
    test_round_1_logic()
