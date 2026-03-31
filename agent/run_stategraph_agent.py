from embedder.embedder_factory import EmbedderFactory
from vectordb.faiss_db import FaissVectorDB
from retriever.financial_retriever import FinancialRetriver
from llm.llm_factory import LLMFactory
from graph.state import State
from graph.node import RetrieverNode, LLMNode, BacktestNode, SearchNode
from graph.utils import route_state
from langgraph.graph import StateGraph, END
from utils.trace_utils import (
    print_stage,
    validate_tool_result,
    validate_llm_output,
    format_tool_result_summary,
    print_contract_check,
)


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
    
    print("=" * 60)
    print("StateGraph Execution Trace")
    print("=" * 60)
    print_stage("stategraph", "start", query=query[:50] + "..." if len(query) > 50 else query)
    print()
    
    # Track execution stages for summary
    stage_results = {}
    final_llm_answer = None
    
    # Use streaming for runtime visibility
    for event in agent.stream({"query": query, "debug": {}}):
        # Each event has one key = node name
        for node_name, node_output in event.items():
            print_stage(node_name, "complete")
            
            # Validate outputs based on node type
            if node_name == "retrieve":
                docs = node_output.get("docs", [])
                print(f"   Retrieved {len(docs)} document(s)")
                # Print actual retrieved docs with text snippets
                for i, doc in enumerate(docs):
                    doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
                    print(f"   --- Doc {i+1}: {doc_preview}")
                
            elif node_name == "route":
                run_search = node_output.get("run_search", False)
                run_backtest = node_output.get("run_backtest", False)
                ticker = node_output.get("ticker", "")
                csv_path = node_output.get("csv_path", "")
                print(f"   Route: search={run_search}, backtest={run_backtest}")
                if ticker:
                    print(f"   Ticker: {ticker}")
                if csv_path:
                    print(f"   CSV Path: {csv_path}")
                
            elif node_name == "search":
                search_results = node_output.get("search_results")
                validation = validate_tool_result(search_results)
                summary = format_tool_result_summary(search_results)
                print(f"   {summary}")
                # Print concrete search results: title, link, snippet
                if search_results and search_results.get("data"):
                    print(f"   --- Search Results ({len(search_results['data'])} items):")
                    for i, item in enumerate(search_results["data"]):
                        title = item.get("title", "N/A")
                        link = item.get("link", "N/A")
                        snippet = item.get("snippet", "")[:100] if item.get("snippet") else ""
                        print(f"   [{i+1}] {title}")
                        print(f"       Link: {link}")
                        if snippet:
                            print(f"       Snippet: {snippet}...")
                print_contract_check("search", validation)
                stage_results["search"] = validation
                
            elif node_name == "backtest":
                backtest = node_output.get("backtest")
                validation = validate_tool_result(backtest)
                summary = format_tool_result_summary(backtest)
                print(f"   {summary}")
                # Print key performance fields and sample trades
                if backtest and backtest.get("data"):
                    perf = backtest.get("data", {}).get("performance", {})
                    if perf:
                        print(f"   --- Performance:")
                        print(f"       Initial Cash: {perf.get('initial_cash', 'N/A')}")
                        print(f"       Final Cash: {perf.get('final_cash', 'N/A')}")
                        print(f"       Return %: {perf.get('return_pct', 'N/A')}")
                        print(f"       Sharpe Ratio: {perf.get('sharpe_ratio', 'N/A')}")
                    trades = backtest.get("data", {}).get("trades", [])
                    if trades:
                        print(f"   --- Sample Trades ({len(trades)} total):")
                        for i, trade in enumerate(trades[:3]):  # Show first 3 trades
                            print(f"       [{i+1}] {trade.get('date')}: {trade.get('action')} {trade.get('size')} @ {trade.get('price')}")
                print_contract_check("backtest", validation)
                stage_results["backtest"] = validation
                
            elif node_name == "llm":
                answer = node_output.get("answer")
                validation = validate_llm_output(answer)
                debug = node_output.get("debug", {})
                if validation["valid"]:
                    verdict = answer.get("verdict", "N/A")
                    recommendation = answer.get("recommendation", "N/A")
                    backtest_summary = answer.get("backtest_summary")
                    print(f"   Verdict: {verdict}")
                    print(f"   Recommendation: {recommendation[:100]}..." if len(recommendation) > 100 else f"   Recommendation: {recommendation}")
                    if backtest_summary:
                        print(f"   Backtest Summary: {backtest_summary}")
                    # Print raw output preview from debug
                    raw_output = debug.get("llm_output_raw", "")
                    if raw_output:
                        raw_preview = raw_output[:150] + "..." if len(raw_output) > 150 else raw_output
                        print(f"   Raw Output Preview: {raw_preview}")
                    # Capture final answer for later display
                    final_llm_answer = answer
                else:
                    print(f"   Raw answer: {str(answer)[:80]}...")
                print_contract_check("llm", validation)
                stage_results["llm"] = validation
                
            print()
    
    print_stage("stategraph", "complete")
    
    # Print contract summary
    print()
    print("=" * 60)
    print("Contract Summary")
    print("=" * 60)
    for stage, result in stage_results.items():
        symbol = "✓" if result["valid"] else "✗"
        print(f"  {symbol} {stage}: {result['message']}")
    
    print()
    print("=" * 60)
    print("Final Answer")
    print("=" * 60)
    # Print final answer captured during streaming loop
    if final_llm_answer:
        print(final_llm_answer)
