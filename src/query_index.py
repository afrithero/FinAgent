from embedder.embedder_factory import EmbedderFactory
from vectordb.faiss_db import FaissVectorDB

if __name__ == "__main__":
    embedder = EmbedderFactory.create_embedder(provider="huggingface", model_name="avsolatorio/GIST-all-MiniLM-L6-v2")
    faiss_db = FaissVectorDB(embed_model=embedder)
    faiss_db.load('../data/FinDER')
    responses = faiss_db.query('Delta in CBOE Data & Access Solutions rev from 2021-23.')
    for res in responses:
        print("Response: ", res.text[:200])

