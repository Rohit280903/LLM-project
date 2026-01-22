import os
import requests
from typing import List


class DeepSeekGenerator:
    def __init__(self):
        """
        Initialize DeepSeek API using environment variable
        """
        self.api_key = os.getenv("sk-875443f482fb488eba1001695c39e008")

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found. Set it as an environment variable."
            )
        # self.api_url = "https://platform.deepseek.com/api_keys"

        self.api_url = "https://platform.deepseek.com/api_keys/v1/chat/completions"

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate answer using retrieved context and user query
        """
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are an AI assistant.
Answer the question strictly using the provided context.
If the answer is not present, say "Information not found in documents".

Context:
{context}

Question:
{query}

Answer:
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
