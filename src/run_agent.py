from embedder.embedder_factory import EmbedderFactory
from vectordb.faiss_db import FaissVectorDB
from retriever.financial_retriever import FinancialRetriver
from agent.tool_factory import ToolFactory
from llm.llm_factory import LLMFactory
from langgraph.graph import StateGraph, END
from graph.state import State
from graph.node import RetrieverNode, LLMNode


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
    llm_node = LLMNode(llm)
    
    graph = StateGraph(State)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("llm", llm_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "llm")
    graph.add_edge("llm", END)
    agent = graph.compile()
    
    query = "Analyze Stock Market of 2024."
    response = agent.invoke({"query": query, "debug": {}})
    print("\n========== RAG Pipeline Debug ==========")
    print(f"Query:\n{query}\n")

    print("----- Retriever Output -----")
    print(response["debug"].get("retriever_output", ""))

    print("\n----- LLM Raw Output -----")
    print(response["debug"].get("llm_output", ""))