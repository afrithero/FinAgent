from datasets import load_dataset
from llama_index.core import Document
from .base_dataset import BaseDataset

class FinderDataset(BaseDataset):
    def __init__(self, split="train"):
        super().__init__(split)
    
    def load(self):
        ds = load_dataset("Linq-AI-Research/FinDER")
        data = ds[self.split]

        return [
            Document(text=f"Question: {q}\nAnswer: {a}")
            for q, a in zip(data["text"], data["answer"])
        ]