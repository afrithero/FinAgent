from .hf_llm import HFLLM

class LLMFactory:
    @staticmethod
    def create_llm(provider, model_name):
        if provider == "huggingface":
            return HFLLM(model_name)
        else:
            raise ValueError(f"Currently support HuggingFace only.")