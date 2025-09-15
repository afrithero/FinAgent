from abc import ABC, abstractmethod

class BaseDataset(ABC):
    def __init__(self, split="train"):
        self.split = split
        
    @abstractmethod
    def load(self):
        return NotImplementedError