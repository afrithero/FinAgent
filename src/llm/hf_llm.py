from .base_llm import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class HFLLM(BaseLLM):
    def __init__(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer,  device_map="auto")
    
    def generate(self, prompt):
        response = self.generator(prompt)[0]["generated_text"]
        return response