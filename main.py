import os
from pipeline_orchestrator import MRCDPipeline

def main():
    print("=== Multi-Round Co-Evolution Detection (MRCD) Framework ===")
    
    # Check for API Key
    api_key_status = "Found" if os.environ.get("GEMINI_API_KEY") else "Not Found (Will use Mock)"
    print(f"Environment Check: GEMINI_API_KEY -> {api_key_status}")
    
    # Initialize Pipeline
    # Default to 'gemini' if key exists, else 'mock'
    llm_mode = "gemini" if os.environ.get("GEMINI_API_KEY") else "mock"
    pipeline = MRCDPipeline(llm_type=llm_mode)
    
    # Target Event Stream (X_t): Mix of potential Fake/Real news
    emergent_events = [
        "NASA announces the discovery of a new planet in the habitable zone of Proxima Centauri with potential liquid water.", 
        "Government to impose a 50% tax on all breathing air starting next month due to pollution.", # Obvious Fake
        "Local cat elected mayor of small Alaskan town after human candidates disqualify.", # Sounds fake but might be real (Talkeetna!)
        "New study shows that eating chocolate every day guarantees weight loss.", # Fake medical claim
        "Tech giant Apple releases the new iPhone 16 with a holographic display feature.", # Probably fake/rumor
        "Breaking: Major earthquake strikes the coast of Japan, measuring 7.5 on the Richter scale.", # Real-style event
        "Scientists have successfully transmitted solar power from space to Earth for the first time." # Tech advance
    ]
    
    print(f"\nLoaded {len(emergent_events)} emergent events for processing.")
    
    # Run the Pipeline
    # The 'run_batch' method separates Clean vs Noisy and runs the Multi-Round loop
    final_results = pipeline.run_batch(emergent_events)
    
    # Display Results
    print("\n" + "="*50)
    print("FINAL FRAMEWORK RESULTS")
    print("="*50)
    
    real_count = 0
    fake_count = 0
    
    for i, item in enumerate(final_results):
        # 1 = Fake, 0 = Real
        label_str = "FAKE" if item['label'] == 1 else "REAL"
        if item['label'] == 1: fake_count += 1
        else: real_count += 1
            
        print(f"{i+1}. [{label_str}] (Conf: {item['confidence']:.2f}, Src: {item['source']})")
        print(f"   Text: {item['text'][:80]}...")
        # print("-" * 30)
        
    print(f"\nSummary: {real_count} Real, {fake_count} Fake identified.")

if __name__ == "__main__":
    main()
