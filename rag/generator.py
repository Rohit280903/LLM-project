from groq import Groq
import os
from typing import List

class GroqGenerator:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)

        prompt = f"""
Answer using only the context below.
If not found, say "Information not found in documents".

Context:
{context}

Question:
{query}
"""

        response = self.client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)


        return response.choices[0].message.content
