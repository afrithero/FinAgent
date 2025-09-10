from abc import ABC, abstractmethod

class BaseDataset(ABC):
    @abstractmethod
    def load(self):
        pass