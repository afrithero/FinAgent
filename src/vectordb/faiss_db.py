import faiss
from .base_db import BaseVectorDB
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage 
from llama_index.core.node_parser import SentenceSplitter

class FaissVectorDB(BaseVectorDB):
    def __init__(self, embed_model):
        self.embed_model = embed_model
        test_vec = self.embed_model.get_text_embedding("test")
        self.dim = len(test_vec)
        self.index = None 
        self.vector_store = None
        self.storage_context = None
    
    def build_index(self, docs, chunk_size, batch_size):
        faiss_index = faiss.IndexFlatL2(self.dim)
        splitter = SentenceSplitter(chunk_size=chunk_size)
        self.vector_store = FaissVectorStore(faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(
            docs, 
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            batch_size=batch_size,
            transformations=[splitter],
            show_progress=True)

        return self.index
    
    def persist(self, path):
        if self.index:
            self.index.storage_context.persist(path)
    
    def load(self, path):
        self.vector_store = FaissVectorStore.from_persist_dir(path)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store, 
            persist_dir=path
        )
        self.index = load_index_from_storage(
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )
    
    def query(self, text, top_k=3):
        # query_engine = self.index.as_query_engine()
        # response = query_engine.query(text)
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        responses = retriever.retrieve(text)
        return responses
