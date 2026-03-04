import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from abc import ABC, abstractmethod
from typing import Optional
from google import genai
from google.genai import types

class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response."""
        pass


class MockLLM(BaseLLM):
    """Mock LLM for testing without costs."""
    
    def generate_text(self, prompt: str) -> str:
        return "This is a mock response for testing purposes."

class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        # Use provided key or fallback to environment variable
        _api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not _api_key:
            print("Warning: GEMINI_API_KEY is not set.")
        
        # Initialize the client with the API key
        self.client = genai.Client(api_key=_api_key)
    
    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=types.Part.from_text(text=prompt)
            )
            # Adjust based on the actual response structure of google.genai library
            # Usually it's response.text or response.candidates[0].content.parts[0].text
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"

# --- Factory / Singleton Management ---

_current_llm: Optional[BaseLLM] = None

def get_llm() -> BaseLLM:
    """Get the currently configured LLM instance."""
    global _current_llm
    if _current_llm is None:
        # Default to a safe option or raise error
        print("LLM not configured, using MockLLM.")
        _current_llm = MockLLM()
    return _current_llm

def set_llm(llm_instance: BaseLLM):
    """Set the global LLM instance."""
    global _current_llm
    _current_llm = llm_instance

def call_llm(prompt: str) -> str:
    """
    Main entry point for other modules.
    Delegates to the configured LLM instance.
    """
    return get_llm().generate_text(prompt)

if __name__ == "__main__":
    # Example usage
    set_llm(GeminiLLM(model_name="gemini-2.5-flash"))
    response = call_llm("What is the capital of France?")
    print(response)