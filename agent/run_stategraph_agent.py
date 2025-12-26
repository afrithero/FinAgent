from embedder.embedder_factory import EmbedderFactory
from vectordb.faiss_db import FaissVectorDB
from retriever.financial_retriever import FinancialRetriver
from llm.llm_factory import LLMFactory
from graph.state import State
from graph.node import RetrieverNode, LLMNode, BacktestNode, SearchNode
from graph.utils import route_state
from langgraph.graph import StateGraph, END


if __name__ == "__main__":
    embedder = EmbedderFactory.create_embedder(
        provider="huggingface", 
        model_name="avsolatorio/GIST-all-MiniLM-L6-v2")
    
    llm = LLMFactory.create_llm(
        provider="huggingface",
        model_name="deepseek-ai/DeepSeek-R1-0528",
        temperature=0
    )
    
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
    
    def _decide_next(state):
        if state.get("run_search"):
            return "search"
        return "backtest" if state.get("run_backtest") else "llm"

    def _decide_after_search(state):
        return "backtest" if state.get("run_backtest") else "llm"

    graph = StateGraph(State)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("route", route_state)
    graph.add_node("search", SearchNode())
    graph.add_node("backtest", backtest_node)
    graph.add_node("llm", llm_node)
    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "route")
    graph.add_conditional_edges(
        "route",
        _decide_next,
        {
            "search": "search",
            "backtest": "backtest",
            "llm": "llm",
        },
    )
    graph.add_conditional_edges(
        "search",
        _decide_after_search,
        {
            "backtest": "backtest",
            "llm": "llm",
        },
    )
    graph.add_edge("backtest", "llm")
    graph.add_edge("llm", END)
    agent = graph.compile()
    
    query = "I have already based on stock price data Apple stock (AAPL) of since January 2023 and used an SMA cross strategy for backtesting. Does Apple stock (AAPL) have growth potential in the market?"
    response = agent.invoke({"query": query, "debug": {}})
    print("----- LLM Answer -----")
    print(response["answer"])
