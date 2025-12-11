from .base_llm import BaseLLM
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
    
class HFLLM(BaseLLM):
    def __init__(self, repo_id, temperature):
        self.chat_model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=repo_id,
                temperature=temperature,
            )
        )

    def generate(self, prompt):
        resp = self.chat_model.invoke([HumanMessage(content=prompt)])
        return resp.content

    def bind_tools(self, tools):
        return self.chat_model.bind_tools(tools)
