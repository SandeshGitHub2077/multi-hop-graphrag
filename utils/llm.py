import os
from typing import Optional
from langchain_ollama import ChatOllama


class LLMWrapper:
    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://127.0.0.1:11434",
    ):
        self.model = model
        self.base_url = base_url
        self.llm: Optional[ChatOllama] = None

    def load(self):
        self.llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0.1,
        )
        print(f"Loaded LLM: {self.model}")

    def generate(self, prompt: str) -> str:
        if self.llm is None:
            self.load()
        response = self.llm.invoke(prompt)
        return response.content

    def generate_with_context(self, query: str, context: str) -> str:
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        return self.generate(prompt)
