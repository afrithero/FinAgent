from .tools import backtest_tool
import json

class RetrieverNode:
    def __init__(self, retriever, top_k=1):
        self.retriever = retriever
        self.top_k = top_k

    def __call__(self, state):
        docs = self.retriever.retrieve(state["query"])
        selected_docs = docs[:self.top_k]
        debug = state.get("debug", {}).copy()
        debug["retriever_output"] = "\n---\n".join(selected_docs) if selected_docs else "[No docs]"
        
        return {
            "docs": selected_docs, 
            "debug": debug
            }

class BacktestNode:
    def __init__(self, csv_path, cash, fast, slow):
        self.csv_path = csv_path
        self.cash = cash
        self.fast = fast
        self.slow = slow
    
    def __call__(self, state):
        result = backtest_tool.invoke({
            "csv_path": self.csv_path,
            "cash": self.cash,
            "fast": self.fast,
            "slow": self.slow
        })

        debug = state.get("debug", {}).copy()
        debug["backtest_output"] = str(result)
        
        return {
            "docs": state["docs"],
            "backtest": result,
            "debug": debug
        }

class LLMNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state):
        context = "\n".join(state["docs"])
        backtest_result = ""
        if "backtest" in state and state["backtest"]:
            backtest_result = json.dumps(state["backtest"], indent=2)
        prompt = f"""
        You are a financial assistant. Please analyze the stock growth potential in the market.

        Context:
        {context}

        Backtest Result:
        {backtest_result}

        Question:
        {state["query"]}

        """
        answer = self.llm.generate(prompt)
        debug = state.get("debug", {}).copy()
        debug["llm_input"] = prompt.strip()
        debug["llm_output"] = answer.strip()

        return {
            "answer": answer,
            "debug": debug
        }
        


    