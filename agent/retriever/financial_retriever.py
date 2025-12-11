from .base_retriever import BaseRetriever

class FinancialRetriver(BaseRetriever):
    def __init__(self, db):
        super().__init__(db)
    
    def retrieve(self, text):
        results = self.db.query(text)
        return [res.text for res in results]