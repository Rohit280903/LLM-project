import os
import re
from typing import List, Generator

try:
    from groq import Groq
except Exception:
    class _ChatCompletions:
        def create(self, model, messages, temperature=0.0, stream=False):
            raise RuntimeError("Install groq: pip install groq")
    class _Chat:
        def __init__(self):
            self.completions = _ChatCompletions()
    class Groq:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.chat = _Chat()


class GroqGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Add it to your .env file or Streamlit Secrets."
            )
        self.client = Groq(api_key=api_key)
        self.qa_model = "llama-3.1-8b-instant"
        self.summary_model = "llama-3.1-8b-instant"

    # ---------------------------
    # Query Rewriting
    # ---------------------------
    def rewrite_query(self, query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.qa_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rewrite this question to be precise and optimized "
                        f"for retrieving relevant information from a document.\n\n"
                        f"Question: {query}\n\nRewritten question:"
                    )
                }],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query  # Fall back to original query on failure

    # ---------------------------
    # Grounded Answer — with streaming + conversation memory
    # ---------------------------
    def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[str],
        chat_history: List[tuple] = None
    ) -> Generator:
        """
        Stream the answer token by token.
        Use with Streamlit's st.write_stream().

        Args:
            query: User's question
            context_chunks: Retrieved document chunks
            chat_history: List of (question, answer) tuples from previous turns
        """
        labeled_context = "\n\n".join(
            f"[Chunk {i}] {chunk}" for i, chunk in enumerate(context_chunks, 1)
        )

        # Build conversation memory string (last 4 turns)
        memory = ""
        if chat_history:
            recent = chat_history[-4:]
            for q, a in recent:
                memory += f"User: {q}\nAssistant: {a}\n\n"

        system_prompt = (
            "You are a document-grounded AI assistant.\n"
            "RULES:\n"
            "- Answer ONLY using the provided context.\n"
            "- Do NOT use external knowledge.\n"
            "- If the answer is not present, say: "
            "'Information not found in the uploaded documents.'\n"
            "- Mention the chunk numbers you used at the end of your answer."
        )

        messages = [{"role": "system", "content": system_prompt}]

        if memory:
            messages.append({
                "role": "user",
                "content": f"Previous conversation:\n{memory}"
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I'll use that context for follow-up questions."
            })

        messages.append({
            "role": "user",
            "content": f"Context:\n{labeled_context}\n\nQuestion: {query}"
        })

        stream = self.client.chat.completions.create(
            model=self.qa_model,
            messages=messages,
            temperature=0.2,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def generate_answer(
        self,
        query: str,
        context_chunks: List[str],
        chat_history: List[tuple] = None
    ) -> str:
        """Non-streaming version. Returns full answer as string."""
        return "".join(
            self.generate_answer_stream(query, context_chunks, chat_history)
        )

    # ---------------------------
    # Hierarchical Summarization (with chunk cap)
    # ---------------------------
    def summarize_document(
        self, context_chunks: List[str], max_chunks: int = 30
    ) -> str:
        """
        Summarize document using batched partial summaries.
        Capped at max_chunks to avoid excessive API calls.
        """
        # Sample evenly if over limit
        if len(context_chunks) > max_chunks:
            step = len(context_chunks) // max_chunks
            context_chunks = context_chunks[::step][:max_chunks]

        partial_summaries = []

        for i in range(0, len(context_chunks), 5):
            batch = context_chunks[i:i + 5]
            context = "\n\n".join(batch)

            try:
                response = self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Summarize the following content in concise bullet points. "
                            f"Focus only on key ideas.\n\nContent:\n{context}\n\nSummary:"
                        )
                    }],
                    temperature=0.3,
                )
                partial_summaries.append(
                    response.choices[0].message.content.strip()
                )
            except Exception as e:
                partial_summaries.append(f"[Batch {i//5 + 1} failed: {e}]")

        combined_text = "\n\n".join(partial_summaries)

        try:
            final_response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Combine these summaries into a single, clear document summary.\n\n"
                        f"Summaries:\n{combined_text}\n\nFinal Summary:"
                    )
                }],
                temperature=0.3,
            )
            return final_response.choices[0].message.content.strip()
        except Exception as e:
            return combined_text  # Return partial summaries if final call fails

    # ---------------------------
    # Suggested Questions (with robust parsing)
    # ---------------------------
    def generate_questions(self, context_chunks: List[str]) -> List[str]:
        context = "\n\n".join(context_chunks[:5])

        try:
            response = self.client.chat.completions.create(
                model=self.qa_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Based on the following document content, generate 5 useful "
                        f"questions a student might ask. Output one question per line.\n\n"
                        f"Content:\n{context}\n\nQuestions:"
                    )
                }],
                temperature=0.4,
            )
            questions_text = response.choices[0].message.content.strip()

            questions = []
            for line in questions_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove leading bullets, numbers, dashes (e.g. "1. ", "- ", "• ")
                line = re.sub(r"^[\d\.\-\•\*\s]+", "", line).strip()
                if line:
                    questions.append(line)

            return questions[:5]  # Return at most 5

        except Exception:
            return []