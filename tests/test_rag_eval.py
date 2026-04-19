from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision
)
from datasets import Dataset

# ✅ Add LLM
from langchain_groq import ChatGroq
import pytest


def test_rag_quality():
    # Initialize LLM
    llm = ChatGroq(model="llama-3.1-8b-instant")

    data = {
        "question": [
            "What is machine learning?",
            "What is RAG?"
        ],
        "answer": [
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "RAG stands for Retrieval Augmented Generation."
        ],
        "contexts": [
            ["Machine learning is a subset of artificial intelligence..."],
            ["RAG combines retrieval with generation to ground LLM answers..."]
        ],
        "ground_truth": [
            "Machine learning allows computers to learn without being explicitly programmed.",
            "RAG is a technique that retrieves relevant documents before generating an answer."
        ]
    }

    dataset = Dataset.from_dict(data)

    def make_dataset():
        data = {
            "question": [
                "What is machine learning?",
                "What is RAG?"
            ],
            "answer": [
                "Machine learning is a subset of AI that enables systems to learn from data.",
                "RAG stands for Retrieval Augmented Generation."
            ],
            "contexts": [
                ["Machine learning is a subset of artificial intelligence..."],
                ["RAG combines retrieval with generation to ground LLM answers..."]
            ],
            "ground_truth": [
                "Machine learning allows computers to learn without being explicitly programmed.",
                "RAG is a technique that retrieves relevant documents before generating an answer."
            ]
        }
        return Dataset.from_dict(data)


    def test_rag_quality():
        dataset = make_dataset()

        result = evaluate(
            dataset,
            metrics=[
                ContextPrecision()
            ]
        )

        print("\nFull Result:", result)

        # ✅ ADD THIS (RAG Accuracy)
        rag_accuracy = sum(result.values()) / len(result)
        print(f"🎯 RAG Accuracy: {rag_accuracy:.2f}")

        assert "context_precision" in result
        assert 0.0 <= result["context_precision"] <= 1.0


    def test_collections_llm_validation():
        llm = ChatGroq(model="llama-3.1-8b-instant")
        dataset = make_dataset()

        with pytest.raises(ValueError, match="Collections metrics only support modern InstructorLLM"):
            evaluate(
                dataset,
                metrics=[
                    Faithfulness(llm=llm),
                    AnswerRelevancy(llm=llm)
                ]
            )


    def test_dataset_structure():
        dataset = make_dataset()
        assert set(dataset.column_names) >= {"question", "answer", "contexts", "ground_truth"}
        assert len(dataset) == 2


if __name__ == "__main__":
    test_rag_quality()
    