from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    def __init__(self, db):
        self.db = db

    @abstractmethod
    def retrieve(self, text):
        return NotImplementedError