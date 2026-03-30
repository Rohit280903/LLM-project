from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

def test_rag_quality():
    # Sample QA pairs from your docs
    data = {
        "question": ["What is machine learning?"],
        "answer": ["Machine learning is..."],        # your RAG answer
        "contexts": [["ML is a subset of AI..."]],  # retrieved chunks
        "ground_truth": ["Machine learning enables machines to learn from data"]
    }
    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

    # Fail CI if quality drops below threshold
    assert result["faithfulness"] > 0.7, "Faithfulness too low!"
    assert result["answer_relevancy"] > 0.7, "Relevancy too low!"