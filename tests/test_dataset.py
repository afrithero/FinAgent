import pytest
from datasets import load_dataset, Value

@pytest.fixture(scope="session")
def ds():
    return load_dataset("Linq-AI-Research/FinDER")

class TestFinderDataset:
    def train_exists(self, ds):
        assert "train" in ds
        assert len(ds["train"]) > 0
    
    def test_schema_columns(self, ds):
        cols = set(ds["train"].column_names)
        required = {"text", "answer", "reasoning"}
        assert required.issubset(cols)
    
    def test_dtypes_and_missing(self, ds):
        train = ds["train"]
        feats = train.features
        assert (isinstance(feats["text"], Value)) and (feats["text"].dtype == "string")
        assert (isinstance(feats["answer"], Value)) and (feats["answer"].dtype == "string")
        assert (isinstance(feats["reasoning"], Value)) and (feats["reasoning"].dtype == "bool")