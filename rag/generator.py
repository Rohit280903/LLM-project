import os
from typing import List

# ---------------------------
# Groq Import with Safe Stub
# ---------------------------
try:
    from groq import Groq
except Exception:
    class _ChatCompletions:
        def create(self, model, messages, temperature=0.0):
            raise RuntimeError(
                "The 'groq' package is not installed. "
                "Install it using: pip install groq"
            )

    class _Chat:
        def __init__(self):
            self.completions = _ChatCompletions()

    class Groq:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.chat = _Chat()


# ---------------------------
# Generator Class
# ---------------------------
class GroqGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key)

        # Models
        self.qa_model = "llama-3.1-8b-instant"
        self.summary_model = "llama-3.1-8b-instant"

    # ---------------------------
    # Query Rewriting (Agent)
    # ---------------------------
    def rewrite_query(self, query: str) -> str:
        prompt = f"""
Rewrite the following question to be precise and optimized
for retrieving relevant information from a document.

Question:
{query}

Rewritten question:
"""
        response = self.client.chat.completions.create(
            model=self.qa_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        return response.choices[0].message.content.strip()

    # ---------------------------
    # Grounded Answer Generator
    # ---------------------------
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        labeled_context = ""
        for i, chunk in enumerate(context_chunks, 1):
            labeled_context += f"[Chunk {i}] {chunk}\n\n"

        prompt = f"""
You are a document-grounded AI assistant.

RULES:
- Answer ONLY using the provided context.
- Do NOT use external knowledge.
- If the answer is not present, say:
  "Information not found in the uploaded documents."
- Mention chunk numbers used at the end.

Context:
{labeled_context}

Question:
{query}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.qa_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    # ---------------------------
    # SAFE Hierarchical Summarization
    # ---------------------------
    def summarize_document(self, context_chunks: List[str]) -> str:
        """
        Token-safe document summarization using chunk-wise summarization
        """

        partial_summaries = []

        # Summarize in small batches (prevents 413 error)
        for i in range(0, len(context_chunks), 5):
            batch = context_chunks[i:i + 5]
            context = "\n\n".join(batch)

            prompt = f"""
Summarize the following content in concise bullet points.
Focus only on key ideas.

Content:
{context}

Summary:
"""
            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            partial_summaries.append(
                response.choices[0].message.content.strip()
            )

        # Combine partial summaries
        combined_text = "\n\n".join(partial_summaries)

        final_prompt = f"""
Combine the following summaries into a single,
clear and concise document summary.

Summaries:
{combined_text}

Final Summary:
"""

        final_response = self.client.chat.completions.create(
            model=self.summary_model,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.3,
        )

        return final_response.choices[0].message.content.strip()

    # ---------------------------
    # Suggested Questions
    # ---------------------------
    def generate_questions(self, context_chunks: List[str]) -> List[str]:
        context = "\n\n".join(context_chunks[:5])

        prompt = f"""
Based on the following document content,
generate 5 useful questions a student might ask.

Content:
{context}

Questions:
"""

        response = self.client.chat.completions.create(
            model=self.qa_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )

        questions_text = response.choices[0].message.content.strip()

        return [
            q.strip("- ").strip()
            for q in questions_text.split("\n")
            if q.strip()
        ]
