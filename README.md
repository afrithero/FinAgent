# Financial Agent Assistant

This repository is about developing a Financial Agent Assistant, combining financial data RAG, search engine, and Backtrader quantitative trading tools to generate investment recommendation reports.

## Features
- **Vector Database (LlamaIndex + Faiss)**     
  -Build a retrieval system for financial documents (e.g., earnings reports, analyst research, news). Currently experimenting with vector database construction using the FinDER dataset [1] to enable Retrieval-Augmented Generation (RAG) for financial QA.

- **Search Engine API Integration** **(WIP)**         
  Get the latest market news and price updates.

- **Quantitative Backtesting (Backtrader)** **(WIP)**   
  Run trading strategies (e.g., moving averages, momentum) with historical data.

- **AI Agent (LangGraph)** **(WIP)**   
  Intergrates tools (retriever, search, backtester, LLM) to generate comprehensive investment advises.


## Usage

### Install dependencies
```bash
pip install -r requirements.txt
```
### Build the vector database
```bash
cd ./src
python build_index.py
```
This step will:
- Load the FinDER dataset
- Perform chunking and embeddings
- Build a Faiss vector database
- Persist the index to ../data/FinDER/

### Query the vector database
```bash
cd ./src
python query_index.py
```
This step will:
- Load the persisted vector database
- Use a retriever to perform semantic search
- Return the most relevant answer snippets

### Run tests
```bash
cd ./tests
pytest -q
```

## References
[1] Chanyel Choi, Jihoon Kwon, Jaeseon Ha, Hojun Choi, Chaewoon Kim, Yongjae Lee, Jy-yong Sohn, Alejandro Lopez-Lira.  
**FinDER: Financial Dataset for Question Answering and Evaluating Retrieval-Augmented Generation.**  
arXiv:2504.15800 (2025).  
[https://arxiv.org/abs/2504.15800](https://arxiv.org/abs/2504.15800)
