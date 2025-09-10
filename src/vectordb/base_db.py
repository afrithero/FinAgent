from abc import ABC, abstractmethod

class BaseVectorDB(ABC):

    @abstractmethod
    def build_index(self, docs):
        pass

    @abstractmethod
    def persist(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass
    
    @abstractmethod
    def query(self, text, top_k=3):
        pass