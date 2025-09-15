import pytest
import tempfile
import shutil
from vectordb.faiss_db import FaissVectorDB
from llama_index.core import Document, MockEmbedding

@pytest.fixture(scope="function")
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

class TestFaissDB:
    def test_build_and_query(self, tmp_dir):
        embedder = MockEmbedding(embed_dim=1024)
        db = FaissVectorDB(embed_model=embedder, path=str(tmp_dir))
        docs = [
            Document(text="hello"),
            Document(text="world")
        ]
        db.build_index(docs, chunk_size=512, batch_size=64)
        db.persist(tmp_dir)
        db2 = FaissVectorDB(embed_model=embedder, path=str(tmp_dir))
        db2.load()
        results = db2.query("hello", top_k=1)
        print(results)
        assert len(results) == 1
        assert any("hello" in str(r) for r in results)
    