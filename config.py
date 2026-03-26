import os
import logging
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_community.chat_models import ChatOllama

load_dotenv()

class Config:
    """A centralized configuration class for the application."""
    
    # LLM Provider Settings
    ALLOWED_PROVIDERS = ["google", "cohere", "ollama"]
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    # Model Names
    GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
    COHERE_MODEL = "command-a-03-2025"
    OLLAMA_MODEL = "gemma:2b" # Default model

    @staticmethod
    def get_llm(provider="google", model_name=None, google_api_key=None, cohere_api_key=None):
        """Factory function to get the appropriate LLM based on the provider and model name."""
        if provider not in Config.ALLOWED_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        if provider == "google":
            api_key = google_api_key or Config.GOOGLE_API_KEY
            if not api_key:
                raise ValueError("Google API key not provided and GOOGLE_API_KEY not found in .env file")
            model_to_use = model_name or Config.GEMINI_MODEL
            return ChatGoogleGenerativeAI(model=model_to_use, google_api_key=api_key)

        elif provider == "cohere":
            api_key = cohere_api_key or Config.COHERE_API_KEY
            if not api_key:
                raise ValueError("Cohere API key not provided and COHERE_API_KEY not found in .env file")
            model_to_use = model_name or Config.COHERE_MODEL
            return ChatCohere(model=model_to_use, cohere_api_key=api_key)
        
        elif provider == "ollama":
            model_to_use = model_name or Config.OLLAMA_MODEL
            return ChatOllama(model=model_to_use)

def setup_logging():
    """Configures logging to file and console, and adds an exception hook."""
    logger = logging.getLogger()
    # Clear existing handlers to prevent duplicate logs with Flask's reloader
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler('app.log', mode='w')
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add a global exception hook to log uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception