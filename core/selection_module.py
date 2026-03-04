"""Selection module: split data into clean and noisy subsets using agreement between LLM and SLM."""

from models.llm_handler import BaseLLM, GeminiLLM
from models.slm_handler import DummySLM

CONFIDENCE_THRESHOLD = 0.8

def construct_prompt(text_x, demonstrations_D, knowledge_K):
    """
    Constructs the prompt for the LLM.
    
    Args:
        text_x (str): The news text to classify.
        demonstrations_D (list): List of demonstrations (dicts with 'text', 'label').
        knowledge_K (str): Background knowledge retrieved from Wikipedia/external sources.
        
    Returns:
        str: The full prompt string.
    """
    prompt = f"""You are an advanced AI fake news detector.
    
BACKGROUND KNOWLEDGE:
{knowledge_K}

INSTRUCTIONS:
Classify the following news article as "Real" (0) or "Fake" (1).
Use the provided knowledge and the style of the examples below to make your decision.

EXAMPLES:
"""
    for i, demo in enumerate(demonstrations_D):
        # Convert label to 0/1 for consistent examples in prompt if needed, 
        # or keep text if LLM understands better. 
        # Mapping: Authentic/Real/Reliable -> 0, Hoax/Fake/Dubious -> 1
        label_str = demo.get('label', 'Unknown')
        prompt += f"\nExample {i+1}:\nText: {demo.get('text', '')[:200]}...\nLabel: {label_str}\n"

    prompt += f"""
TARGET ARTICLE:
Text: {text_x}

OUTPUT FORMAT:
Return ONLY a single digit: 0 for Real, 1 for Fake. Do not add any explanation.
"""
    return prompt

def parse_llm_response(response_text):
    """
    Parses the LLM response to an integer label (0 or 1).
    Handles potential extra text if the LLM is chatty.
    """
    clean_text = response_text.strip().lower()
    if '0' in clean_text or 'real' in clean_text or 'true' in clean_text:
        return 0
    if '1' in clean_text or 'fake' in clean_text or 'hoax' in clean_text:
        return 1
    # Default fallback if unclear - treat as Fake/Noisy to be safe? Or None?
    # For this mock flow, let's default to 1 (Fake) if unsure.
    return 1

def run_selection_pipeline(text_x, demonstrations_D, knowledge_K, llm: BaseLLM = None, slm = None):
    """
    Executes Step 3: Round 1 Classification & Selection.
    
    Args:
        text_x (str): The input news text.
        demonstrations_D (list): Retrieved demonstrations.
        knowledge_K (str): Retrieved knowledge.
        llm (BaseLLM): The LLM instance.
        slm: The SLM instance (DummySLM or Real).
        
    Returns:
        dict: containing 'label_llm', 'label_slm', 'confidence_slm', 'category' (clean/noisy)
    """
    if llm is None:
        llm = GeminiLLM() # Default to Gemini
    if slm is None:
        slm = DummySLM()

    # Module 3.1: LLM Prediction
    prompt = construct_prompt(text_x, demonstrations_D, knowledge_K)
    print(f"\n--- LLM Prompt ---\n{prompt[:300]}...\n[...truncated...]\n")
    
    llm_raw_response = llm.generate_text(prompt)
    y_hat_1 = parse_llm_response(llm_raw_response)
    print(f"LLM Prediction (y^1): {y_hat_1} (Raw: {llm_raw_response.strip()})")

    # Module 3.2: SLM Prediction
    y_hat_2, p_y_hat_2 = slm.inference(text_x)
    print(f"SLM Prediction (y^2): {y_hat_2}, Confidence p(y^2): {p_y_hat_2:.4f}")

    # Module 3.3: Selection Logic
    # D_clean if: (y^1 == y^2) AND (p(y^2) >= 0.8)
    # D_noisy otherwise
    
    is_agreement = (y_hat_1 == y_hat_2)
    is_confident = (p_y_hat_2 >= CONFIDENCE_THRESHOLD)
    
    if is_agreement and is_confident:
        category = "clean"
    else:
        category = "noisy"
        
    print(f"Selection Result: {category.upper()} (Agreement: {is_agreement}, Confident: {is_confident})")
    
    return {
        "text": text_x,
        "y_hat_1": y_hat_1,
        "y_hat_2": y_hat_2,
        "confidence": p_y_hat_2,
        "category": category
    }

def split_data(dataset):
    """Return D_clean and D_noisy from dataset. THIS IS LEGACY/PLACEHOLDER."""
    # This was the old placeholder. For now, just return empty lists as we use the pipeline above.
    return [], []
