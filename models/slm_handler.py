"""Simple LM handler wrapping RoBERTa/FTT models with train and inference."""
import random

class SLMHandler:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        # load model if provided

    def train(self, train_data, epochs: int = 1):
        """Train the SLM on provided data."""
        pass

    def inference(self, inputs):
        """Run inference on inputs and return predictions."""
        pass


class DummySLM(SLMHandler):
    """
    Mock SLM for development and testing.
    Simulates predictions from a trained RoBERTa/FTT model.
    """
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        print("Initialized DummySLM (Mock Model)")
        self.is_trained = False

    def train(self, train_data, epochs: int = 1):
        """Simulate training process."""
        print(f"[DummySLM] Training on {len(train_data)} samples...")
        print(f"[DummySLM] Hyperparameters: Optimizer=AdamW, LR=1e-3, WeightDecay=1e-4, BatchSize=32")
        self.is_trained = True
        print("[DummySLM] Training complete. Confidence intervals boosted.")

    def inference(self, text_x: str):
        """
        Simulate inference on a single text input.
        Returns:
            tuple: (predicted_label, confidence_score)
            - predicted_label: 0 (Real) or 1 (Fake)
            - confidence_score: float between 0.0 and 1.0
        """
        # Randomly predict 0 or 1
        predicted_label = random.choice([0, 1])
        
        # Random confidence score
        # If trained, increase average confidence slightly to simulate better model
        low = 0.7 if self.is_trained else 0.6
        high = 0.99
        confidence_score = random.uniform(low, high)
        
        return predicted_label, confidence_score
