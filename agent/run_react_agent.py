from graph.tools import create_retriever_tool, backtest_tool, search_stock_info
from embedder.embedder_factory import EmbedderFactory
from vectordb.faiss_db import FaissVectorDB
from retriever.financial_retriever import FinancialRetriver
from llm.llm_factory import LLMFactory
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

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
    response = agent.invoke({ "messages": [HumanMessage(content=query)] }) 
    print(f"Final Result: {response['messages'][-1].content} \n")
    print("Step by step execution")
    for message in response['messages']:
        print(message.pretty_repr())
