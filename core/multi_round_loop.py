"""Multi-round learning loop orchestration."""

from core.selection_module import run_selection_pipeline, construct_prompt, parse_llm_response
from retrievers.bm25_matcher import retrieve_demonstrations_from_clean
import math

CONFIDENCE_THRESHOLD = 0.8
NUM_ROUNDS = 3  # N_th

def run_multi_round_learning(d_clean: list, d_noisy: list, knowledge_cache: dict, slm, llm):
    """
    Execute the multi-round training/inference loop (Step 4).
    
    Args:
        d_clean (list): List of clean samples [{"text": "...", "label": 0/1, ...}].
        d_noisy (list): List of noisy samples [{"text": "...", ...}].
        knowledge_cache (dict): Cache mapping text -> knowledge string.
        slm: The SLM instance (DummySLM or Real).
        llm: The LLM instance.
        
    Returns:
        tuple: (final_d_clean, final_d_noisy)
    """
    print(f"\n--- Starting Multi-Round Learning (Step 4, Rounds 2-{NUM_ROUNDS}) ---")
    
    for round_idx in range(2, NUM_ROUNDS + 1):
        print(f"\n=== Round {round_idx} ===")
        print(f"Current Stats: D_clean={len(d_clean)}, D_noisy={len(d_noisy)}")
        
        # B. Fine-Tuning SLM
        print("Training SLM on D_clean...")
        # Hyperparameters (simulated via print for DummySLM)
        slm.train(d_clean, epochs=1) 
        
        # C. Re-evaluating & Updating
        moved_count = 0
        remaining_noisy = []
        
        # Iterate over a copy of d_noisy to allow modification
        for i, sample in enumerate(d_noisy):
            text_x = sample.get("text", "")
            knowledge_K = knowledge_cache.get(text_x, "")
            
            # 1. Demonstration Shift: Retrieve from D_clean
            # If D_clean is small, retrival might return fewer than k=4, handled gracefully.
            new_demonstrations = retrieve_demonstrations_from_clean(text_x, d_clean, k=4)
            
            # 2. LLM Inference (y_hat_1) with new demonstrations
            # Re-construct prompt because demonstrations changed
            prompt = construct_prompt(text_x, new_demonstrations, knowledge_K)
            llm_response = llm.generate_text(prompt)
            y_hat_1 = parse_llm_response(llm_response)
            
            # 3. SLM Inference (y_hat_2) with fine-tuned model
            y_hat_2, p_y_hat_2 = slm.inference(text_x)
            
            # 4. Update Logic
            # Condition: Agreement AND High Confidence
            condition_met = (y_hat_1 == y_hat_2) and (p_y_hat_2 >= CONFIDENCE_THRESHOLD)
            
            if condition_met:
                # Move to D_clean
                sample['label'] = y_hat_1 # Assign the agreed label
                sample['confidence'] = p_y_hat_2
                sample['round_added'] = round_idx
                d_clean.append(sample)
                moved_count += 1
                # print(f"  Sample {i} moved to clean (Label: {y_hat_1}, Conf: {p_y_hat_2:.2f})")
            else:
                # Keep in D_noisy
                remaining_noisy.append(sample)
        
        d_noisy = remaining_noisy
        print(f"Round {round_idx} Complete. Moved {moved_count} samples. New D_clean size: {len(d_clean)}")
        
        if not d_noisy:
            print("D_noisy is empty. Stopping early.")
            break
            
    print("\n--- Multi-Round Learning Finished ---")
    return d_clean, d_noisy
