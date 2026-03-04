
import os
import json
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger()


def log_feedback(
    feedback_path: str,
    question: str,
    answer: str,
    rating: str,
    sources: list,
    model: str,
    query_time: float
) -> None:
    """
    Append a single feedback entry to a JSONL file.

    JSONL (JSON Lines) stores one JSON object per line.

    Each entry contains everything needed to evaluate and improve the pipeline:
    the question, the answer, the rating, what sources were retrieved, which model generated the answer, 
    and when it happened.

    Args:
        feedback_path (str): Path to the JSONL feedback file
        question (str): The user's question
        answer (str): The model's answer
        rating (str): 'thumbs_up' or 'thumbs_down'
        sources (list): Source chunks retrieved for this query
        model (str): LLM model name used
        query_time (float): How long the query took in seconds
    """
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating,
        "sources": [
            {
                "file": s.get("file"),
                "page": s.get("page"),
                "score": s.get("score")
            }
            for s in sources
        ],
        "model": model,
        "query_time": query_time
    }

    with open(feedback_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(f"Feedback logged | Rating: {rating} | Question: '{question[:50]}'")