from embedder.embedder_factory import EmbedderFactory
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from vectordb.faiss_db import FaissVectorDB
from llama_index.core import Document
from dataset.finder_dataset import FinderDataset

if __name__ == "__main__":
    embedder = EmbedderFactory.create_embedder(
        provider="huggingface", 
        model_name="avsolatorio/GIST-all-MiniLM-L6-v2") # Qwen/Qwen3-Embedding-0.6B
    faiss_db = FaissVectorDB(embed_model=embedder)
    dataset = FinderDataset(split="train")
    docs = dataset.load()
    faiss_db.build_index(docs, chunk_size=512, batch_size=64)
    faiss_db.persist("../data/FinDER")

