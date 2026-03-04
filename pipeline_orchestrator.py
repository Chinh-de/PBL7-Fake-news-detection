from models.llm_handler import set_llm, MockLLM, GeminiLLM, get_llm
from models.slm_handler import DummySLM
from retrievers.bing_news_seach import search_news
from retrievers.bm25_matcher import retrieve_demonstrations
from retrievers.wiki_agent import extract_and_summarize
from core.selection_module import run_selection_pipeline
from core.multi_round_loop import run_multi_round_learning

class MRCDPipeline:
    def __init__(self, llm_type="gemini"):
        self.setup_llm(llm_type)
        self.slm = DummySLM()
        self.knowledge_cache = {}

    def setup_llm(self, llm_type):
        if llm_type == "gemini":
            set_llm(GeminiLLM())
            print("LLM: Gemini (Google GenAI)")
        else:
            set_llm(MockLLM())
            print("LLM: Mock (Random Response)")

    def retrieve_context_step_1(self, text_x):
        """Step 1: Retrieval & Pseudo-labeling (Gear 1, 2, 3)"""
        print(f"\n--- Step 1: Retrieval & Context for: '{text_x[:50]}...' ---")
        
        # Gear 1: Demonstration Retrieval (Bing + BM25)
        # In a real scenario, we might use the text_x as query or extract keywords
        # For simplicity, using first few words as query
        query = " ".join(text_x.split()[:10]) 
        print(f"Searching Bing for: '{query}'")
        try:
            bing_news_strings = search_news(query, max_results=5)
            bing_news_items = [{"content": s, "source": "bing"} for s in bing_news_strings]
        except Exception as e:
            print(f"Bing Search failed: {e}. Using empty list.")
            bing_news_items = []
        
        # Gear 2: Pseudo-Labeling is handled inside retrieve_demonstrations
        # If no Bing results, retrieve_demonstrations will rely on static corpus
        demonstrations = retrieve_demonstrations(query, bing_news_items, k=4)
        print(f"Retrieved {len(demonstrations)} demonstrations.")

        # Gear 3: Knowledge Retrieval (Agent)
        print("Extracting entities for Knowledge Retrieval...")
        
        try:
            knowledge_dict = extract_and_summarize(text_x)
            knowledge_texts = [f"{entities}: {summary}" for entities, summary in knowledge_dict.items()]
            knowledge_K = "\n".join(knowledge_texts) if knowledge_texts else "No specific knowledge found."
        except Exception as e:
            print(f"Knowledge Retrieval failed: {e}")
            knowledge_K = "Knowledge retrieval error."
            
        print(f"Knowledge Context Length: {len(knowledge_K)} chars")
        
        # Cache knowledge for Step 4 use
        self.knowledge_cache[text_x] = knowledge_K
        
        return demonstrations, knowledge_K

    def selection_step_2(self, text_x, demonstrations, knowledge_K):
        """Step 2: Round 1 Selection"""
        print("--- Step 2: Round 1 Selection ---")
        result = run_selection_pipeline(
            text_x, 
            demonstrations, 
            knowledge_K, 
            llm=get_llm(), 
            slm=self.slm
        )
        return result

    def multi_round_step_3_4(self, d_clean, d_noisy):
        """Step 3 & 4: Multi-Round Learning & Final Judgment"""
        if not d_noisy:
            print("No noisy samples to process in multi-round loop.")
            return d_clean, []

        print(f"\n--- Step 3: Multi-Round Learning with {len(d_clean)} Clean, {len(d_noisy)} Noisy samples ---")
        final_clean, final_noisy = run_multi_round_learning(
            d_clean, 
            d_noisy, 
            self.knowledge_cache, 
            self.slm, 
            get_llm()
        )
        
        # Step 4: Final Judgment for remaining noisy samples
        print(f"\n--- Step 4: Final Judgment on {len(final_noisy)} remaining samples ---")
        judged_noisy = []
        for sample in final_noisy:
            # Force prediction using trained SLM
            text = sample['text']
            label, conf = self.slm.inference(text)
            sample['label'] = label
            sample['confidence'] = conf
            sample['source'] = "FinalJudgment_SLM"
            judged_noisy.append(sample)
            # print(f"Forced judgment for noisy sample -> Label {label}")
        
        # Combine judged noisy into a result list (but conceptually they are still 'noisy' confidence)
        # But for the pipeline output, we return everything classified.
        return final_clean + judged_noisy

    def process_single_event(self, text_x):
        """Flow for a single emergent event item (Step 1 & 2). Returns classified object."""
        
        # 1. Retrieve
        demos, knowledge = self.retrieve_context_step_1(text_x)
        
        # 2. First Selection
        initial_result = self.selection_step_2(text_x, demos, knowledge)
        
        sample_data = {
            "text": text_x,
            "label": initial_result["y_hat_1"], # Use LLM label if clean, else undefined/ambiguous
            "confidence": initial_result["confidence"],
            "category": initial_result["category"],
            "source": "Round1"
        }
        return sample_data

    def run_batch(self, events: list[str]):
        """
        Orchestrate the full MRCD pipeline on a batch of emergent events.
        """
        d_clean = []
        d_noisy = []
        
        print(f"Processing batch of {len(events)} events...")
        
        # Round 1 Processing (Parallelizable in theory)
        for i, text in enumerate(events):
            print(f"\n[Event {i+1}/{len(events)}]")
            result = self.process_single_event(text)
            
            if result["category"] == "clean":
                d_clean.append(result)
                print(f"-> Classified as Clean (Conf: {result['confidence']:.2f})")
            else:
                d_noisy.append(result)
                print(f"-> Classified as Noisy (Conf: {result['confidence']:.2f})")
        
        # Round 2 & 3 & 4 Processing (The Loop)
        if d_noisy:
            final_results = self.multi_round_step_3_4(d_clean, d_noisy)
        else:
            final_results = d_clean
            
        return final_results

