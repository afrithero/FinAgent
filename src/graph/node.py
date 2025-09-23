from .state import State

class RetrieverNode:
    def __init__(self, retriever, top_k=3):
        self.retriever = retriever
        self.top_k = top_k

    def __call__(self, state):
        docs = self.retriever.retrieve(state["query"])
        selected_docs = docs[:self.top_k]
        debug = state.get("debug", {}).copy()
        debug["retriever_output"] = "\n---\n".join(selected_docs) if selected_docs else "[No docs]"
        
        return {
            "docs": selected_docs, 
            "debug": {
                "retriever_output": debug}
            }

class LLMNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state):
        context = "\n".join(state["docs"])
        prompt = f"""
        You are a financial assistant. 
        Use the following context to answer the question. 
        - Answer in your own words (do not copy text directly). 
        - Provide a concise and clear explanation. 
        - If numbers are involved, highlight them.
        
        Context:
        {context}

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
        


    