from .hf_llm import HFLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class LLMFactory:
    @staticmethod
    def create_llm(provider, model_name, temperature):
        if provider == "huggingface":
            return HFLLM(repo_id=model_name, temperature=temperature)
        elif provider == "google":
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, api_key=os.getenv("GOOGLE_API_KEY"))
        elif provider == "openai":
            return ChatOpenAI(model=model_name, temperature=temperature)
        else:
            raise ValueError(f"Unsupported provider: {provider}")