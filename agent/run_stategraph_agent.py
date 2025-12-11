from embedder.embedder_factory import EmbedderFactory
from vectordb.faiss_db import FaissVectorDB
from retriever.financial_retriever import FinancialRetriver
from llm.llm_factory import LLMFactory
from graph.state import State
from graph.node import RetrieverNode, LLMNode, BacktestNode
from langgraph.graph import StateGraph, END


if __name__ == "__main__":
    embedder = EmbedderFactory.create_embedder(
        provider="huggingface", 
        model_name="avsolatorio/GIST-all-MiniLM-L6-v2")
    
    llm = LLMFactory.create_llm(
        provider="huggingface", 
        model_name="google/gemma-3-1b-it")
    
    db = FaissVectorDB(
        embed_model=embedder, 
        path="../data/FinDER")
    
    db.load()
    retriever = FinancialRetriver(db)
    retriever_node = RetrieverNode(retriever)
    backtest_node = BacktestNode(
        csv_path="../data/us_stock/AAPL.csv",
        cash=50000, 
        fast=3, 
        slow=37
    )
    llm_node = LLMNode(llm)
    
    graph = StateGraph(State)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("backtest", backtest_node)
    graph.add_node("llm", llm_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "backtest")
    graph.add_edge("backtest", "llm")
    graph.add_edge("llm", END)
    agent = graph.compile()
    
    query = "I have already based on stock price data Apple stock (AAPL) of since January 2023 and used an SMA cross strategy for backtesting. Does Apple stock (AAPL) have growth potential in the market?"
    response = agent.invoke({"query": query, "debug": {}})
    print("----- LLM Answer -----")
    print(response["answer"])