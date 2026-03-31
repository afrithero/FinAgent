from graph.tools import create_retriever_tool, backtest_tool, search_stock_info
from embedder.embedder_factory import EmbedderFactory
from vectordb.faiss_db import FaissVectorDB
from retriever.financial_retriever import FinancialRetriver
from llm.llm_factory import LLMFactory
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from utils.trace_utils import (
    print_stage,
    print_react_event,
    validate_tool_result,
    format_tool_result_summary,
    print_contract_check,
)

if __name__ == "__main__":
    embedder = EmbedderFactory.create_embedder(
        provider="huggingface",
        model_name="avsolatorio/GIST-all-MiniLM-L6-v2"
    )
    db = FaissVectorDB(embed_model=embedder, path="../data/FinDER")
    db.load()
    retriever = FinancialRetriver(db)
    retriever_tool = create_retriever_tool(retriever)

    # LLM
    '''
    Provider:
        huggingface - e.g. deepseek-ai/DeepSeek-R1-0528
        google - e.g. gemini-2.0-flash-exp
        openai - e.g. gpt-4o-mini
    '''
    llm = LLMFactory.create_llm(
        provider="huggingface",
        model_name="deepseek-ai/DeepSeek-R1-0528",
        temperature=0
    )

    # Tool list
    tools = [retriever_tool, backtest_tool, search_stock_info]

    # Create ReAct agent
    prompt = (
        "You are a financial assistant.\n"
        "RULES:\n"
        "1) If the question is about a stock's outlook or predictions, ALWAYS call search_stock_info.\n"
        "2) Then call `retrieve_financial_docs` with the user's query.\n"
        "3) If the query mentions AAPL/Apple, then call `backtest` (use the default csv_path if not provided).\n"
        "4) Please give the paramters of backtesting analyze the result.\n"
        "5) After tool observations, synthesize a final, concise answer."
    )

    agent = create_react_agent(model = llm, 
                               tools = tools,
                               name = "financial_agent",
                               prompt = prompt)

    query = "Does Apple stock (AAPL) have growth potential in the market? Please backtest its historical performance and provide an analysis."
    
    print("=" * 60)
    print("ReAct Agent Execution Trace")
    print("=" * 60)
    print_stage("agent", "start", query=query[:50] + "..." if len(query) > 50 else query)
    print()
    
    # Use streaming for runtime visibility
    tool_count = 0
    final_answer = None
    
    for event in agent.stream({"messages": [HumanMessage(content=query)]}):
        print_react_event(event)
        
        # Track tool invocations
        if "tools" in event:
            tool_count += 1
            
        # Capture final answer when available (only if no tool calls - i.e., actual final response)
        if "agent" in event:
            msgs = event.get("agent", {}).get("messages", [])
            if msgs:
                last_msg = msgs[-1]
                msg_type = type(last_msg).__name__
                # Only capture as final when NOT having tool calls - avoids capturing reasoning
                has_tool_calls = getattr(last_msg, "tool_calls", None) or (
                    hasattr(last_msg, "additional_kwargs") and last_msg.additional_kwargs.get("tool_calls")
                )
                if (msg_type == "AIMessage" or (hasattr(last_msg, "type") and last_msg.type == "ai")) and not has_tool_calls:
                    final_answer = getattr(last_msg, "content", str(last_msg))
    
    print()
    print_stage("agent", "complete")
    print(f"   Total tool invocations: {tool_count}")
    
    print()
    print("=" * 60)
    print("Final Answer")
    print("=" * 60)
    print(final_answer if final_answer else "(No final answer captured)")
