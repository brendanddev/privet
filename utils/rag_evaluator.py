"""
utils/rag_evaluator.py

An automatic local RAG evaluation system.

Scores every answer across four metrics without any external API calls:
Faithfulness, answer relevance, context percision, source coverage.
All scores are stored in a local SQLite database for trend tracking.
Zero network calls are made at any point.
"""

import gc
import os
import re
import sqlite3
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import numpy as np
from utils.logger import setup_logger

logger = setup_logger()

DB_PATH = "./logs/eval_scores.db"


## Score result data class
@dataclass
class EvalResult:
    """
    Scores for a single RAG query. All scores are in [0.0, 1.0].
    Higher is always better.
    """
    question: str
    answer: str
    query_time: float
    timestamp: str

    # Core metrics
    faithfulness: float
    answer_relevance: float
    context_precision: float 
    source_coverage: float    

    # Derived
    composite_score: float 
    tier: str  

    # Metadata
    num_sources: int
    avg_source_score: float
    answer_length: int
    faithfulness_detail: str = "" 


## Database layer
class EvalStore:
    """
    Persists evaluation results in a local SQLite database.
    Schema is append-only — nothing is ever deleted automatically.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    query_time REAL,
                    faithfulness REAL,
                    answer_relevance REAL,
                    context_precision REAL,
                    source_coverage REAL,
                    composite_score REAL,
                    tier TEXT,
                    num_sources INTEGER,
                    avg_source_score REAL,
                    answer_length INTEGER,
                    faithfulness_detail TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON eval_results (timestamp)
            """)
            conn.commit()

    def save(self, result: EvalResult):
        """Persist a scored result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO eval_results (
                    timestamp, question, answer, query_time,
                    faithfulness, answer_relevance, context_precision,
                    source_coverage, composite_score, tier,
                    num_sources, avg_source_score, answer_length,
                    faithfulness_detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.timestamp, result.question, result.answer,
                result.query_time, result.faithfulness, result.answer_relevance,
                result.context_precision, result.source_coverage,
                result.composite_score, result.tier, result.num_sources,
                result.avg_source_score, result.answer_length,
                result.faithfulness_detail
            ))
            conn.commit()

    def get_recent(self, n: int = 50) -> list[dict]:
        """Return the n most recent scored results as dicts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM eval_results
                ORDER BY id DESC LIMIT ?
            """, (n,)).fetchall()
        return [dict(r) for r in rows]

    def get_trend(self, days: int = 7) -> list[dict]:
        """Return daily average scores for the last N days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    DATE(timestamp) as day,
                    COUNT(*) as num_queries,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(answer_relevance) as avg_relevance,
                    AVG(context_precision) as avg_precision,
                    AVG(composite_score) as avg_composite,
                    AVG(query_time) as avg_query_time
                FROM eval_results
                WHERE timestamp >= DATE('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY day ASC
            """, (f'-{days} days',)).fetchall()
        return [dict(r) for r in rows]

    def get_summary(self) -> dict:
        """Return overall statistics across all stored results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    AVG(composite_score) as avg_composite,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(answer_relevance) as avg_relevance,
                    AVG(context_precision) as avg_precision,
                    MIN(composite_score) as min_composite,
                    MAX(composite_score) as max_composite,
                    SUM(CASE WHEN composite_score < 0.5 THEN 1 ELSE 0 END) as low_quality_count
                FROM eval_results
            """).fetchone()
        return dict(row) if row else {}

    def get_low_quality(self, threshold: float = 0.5, n: int = 20) -> list[dict]:
        """Return the lowest scoring queries for review."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM eval_results
                WHERE composite_score < ?
                ORDER BY composite_score ASC
                LIMIT ?
            """, (threshold, n)).fetchall()
        return [dict(r) for r in rows]


## Tier 1: Embedding-based scoring (fast, always available)
class EmbeddingScorer:
    """
    Uses sentence embeddings for fast metric computation.

    Requires no additional models — reuses nomic-embed-text via the
    sentence-transformers library which is already a project dependency.
    Falls back to all-MiniLM-L6-v2 (~22MB) if nomic isn't available
    via sentence-transformers directly.

    All operations are CPU-bound and take <50ms per query.
    """

    def __init__(self):
        self._model = None
        self._model_name = None

    def _load(self):
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            # Prefer the small, fast MiniLM for evaluation
            # It's ~22MB and runs entirely on CPU in <10ms per sentence
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._model_name = "all-MiniLM-L6-v2"
            logger.info("EmbeddingScorer loaded: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"EmbeddingScorer failed to load: {e}")
            raise

    def _embed(self, texts: list[str]):
        """Embed a list of texts, returns numpy array."""
        self._load()
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def _cosine(self, a, b) -> float:
        """Cosine similarity between two normalized vectors."""
        return float(np.dot(a, b))

    def score_answer_relevance(self, question: str, answer: str) -> float:
        """
        Measure how well the answer addresses the question.

        Method: embed question and answer separately, compute cosine similarity.
        A higher score means the answer is semantically close to what was asked.

        This is the RAGAS answer relevance approach but without the LLM — we
        skip the "generate questions from the answer" step and use direct
        semantic similarity instead. Simpler but still useful as a signal.

        Returns: float in [0, 1]
        """
        if not answer.strip() or not question.strip():
            return 0.0
        try:
            embeddings = self._embed([question, answer])
            score = self._cosine(embeddings[0], embeddings[1])
            # Cosine between normalized vectors is in [-1, 1], clamp to [0, 1]
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Answer relevance scoring failed: {e}")
            return 0.5   # neutral fallback

    def score_context_precision(self, question: str, contexts: list[str]) -> float:
        """
        Measure the signal-to-noise ratio of the retrieved context.

        Method: embed the question and each context chunk, compute similarity
        of each chunk to the question, return the mean similarity score.

        A low score means the retrieved chunks are not very relevant to the
        question — the retriever is returning noisy results.

        Returns: float in [0, 1]
        """
        if not contexts or not question.strip():
            return 0.0
        try:
            texts = [question] + contexts
            embeddings = self._embed(texts)
            q_embed = embeddings[0]
            ctx_embeddings = embeddings[1:]
            similarities = [self._cosine(q_embed, c) for c in ctx_embeddings]
            return max(0.0, min(1.0, sum(similarities) / len(similarities)))
        except Exception as e:
            logger.warning(f"Context precision scoring failed: {e}")
            return 0.5

    def score_source_coverage(
        self,
        answer: str,
        sources: list[dict]
    ) -> float:
        """
        Estimate how much of the answer is supported by the retrieved sources.

        Method: split the answer into sentences, embed each sentence and each
        source preview, find the max similarity of each answer sentence to any
        source. Return the fraction of sentences that have a source match above
        a similarity threshold.

        This is a proxy for faithfulness that doesn't require NLI — it checks
        whether the semantic content of the answer is present in the sources,
        not whether claims are logically entailed.

        Returns: float in [0, 1]
        """
        if not answer.strip() or not sources:
            return 0.0

        source_texts = [s.get("preview", "") for s in sources if s.get("preview")]
        if not source_texts:
            return 0.5 

        try:
            sentences = _split_sentences(answer)
            if not sentences:
                return 0.0

            all_texts = sentences + source_texts
            embeddings = self._embed(all_texts)

            sent_embeds = embeddings[:len(sentences)]
            src_embeds = embeddings[len(sentences):]

            THRESHOLD = 0.45 
            matched = 0
            for s_embed in sent_embeds:
                max_sim = max(self._cosine(s_embed, src) for src in src_embeds)
                if max_sim >= THRESHOLD:
                    matched += 1

            return matched / len(sentences)
        except Exception as e:
            logger.warning(f"Source coverage scoring failed: {e}")
            return 0.5

## Tier 2: NLI-based faithfulness scoring (accurate, optional)
class FaithfulnessScorer:
    """
    Uses a Natural Language Inference (NLI) cross-encoder to check whether
    each claim in the answer is entailed by the retrieved context.

    Model: cross-encoder/nli-deberta-v3-small
      - Size: ~86MB on disk, ~180MB in RAM
      - Speed: ~20-40ms per (context, claim) pair on Apple Silicon CPU
      - Quality: outperforms larger models on MNLI benchmark

    The approach mirrors RAGAS faithfulness:
      1. Split the answer into atomic claims (sentences as a proxy)
      2. For each claim, check if ANY retrieved context chunk entails it
      3. Faithfulness = fraction of claims supported by context

    The model is loaded lazily and can be unloaded after scoring to free RAM.
    """

    MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

    def __init__(self):
        self._model = None
        self._loaded = False

    def _load(self):
        """Lazy-load the NLI model."""
        if self._loaded:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.MODEL_NAME)
            self._loaded = True
            logger.info(f"FaithfulnessScorer loaded: {self.MODEL_NAME}")
        except Exception as e:
            logger.warning(f"FaithfulnessScorer failed to load: {e}")
            self._model = None
            self._loaded = False

    def unload(self):
        """Unload the model to free ~180MB of RAM."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            gc.collect()
            logger.info("FaithfulnessScorer unloaded to free RAM")

    def is_available(self) -> bool:
        """Check if the NLI model can be loaded."""
        try:
            from sentence_transformers import CrossEncoder
            return True
        except ImportError:
            return False

    def score(self, answer: str, contexts: list[str]) -> tuple[float, str]:
        """
        Score faithfulness of the answer against the retrieved contexts.

        Args:
            answer: The generated answer text
            contexts: List of retrieved context chunk texts

        Returns:
            tuple: (faithfulness_score float, detail_json str)
              faithfulness_score: fraction of claims supported by context
              detail_json: per-claim breakdown for debugging
        """
        if not answer.strip() or not contexts:
            return 0.5, "[]"

        self._load()
        if self._model is None:
            return 0.5, "[]"   # model unavailable, neutral fallback

        claims = _split_sentences(answer)
        if not claims:
            return 1.0, "[]"

        # Concatenate all contexts into one premise (for short answers)
        # For longer contexts, score each claim against the best matching chunk
        combined_context = " ".join(contexts)[:2000]   # cap at 2000 chars

        # Score each claim
        # NLI label mapping: 0=contradiction, 1=entailment, 2=neutral
        pairs = [(combined_context, claim) for claim in claims]

        try:
            scores = self._model.predict(pairs)
            # scores shape: (n_claims, 3) — contradiction, entailment, neutral
            # (DeBERTa NLI uses this label order)

            details = []
            supported = 0

            for i, (claim, score_row) in enumerate(zip(claims, scores)):
                # Get probability of entailment
                probs = self._softmax(score_row)
                # Label order for cross-encoder/nli-deberta-v3-*:
                # 0=contradiction, 1=entailment, 2=neutral
                entail_prob = float(probs[1])
                contradiction_prob = float(probs[0])

                # A claim is "supported" if entailment > 0.5
                is_supported = entail_prob > 0.5
                if is_supported:
                    supported += 1

                details.append({
                    "claim": claim,
                    "entailment": round(entail_prob, 3),
                    "contradiction": round(contradiction_prob, 3),
                    "supported": is_supported
                })

            faithfulness = supported / len(claims)
            return round(faithfulness, 4), json.dumps(details)

        except Exception as e:
            logger.warning(f"NLI scoring failed: {e}")
            return 0.5, "[]"

    @staticmethod
    def _softmax(x) -> list:
        """Compute softmax over a list of logits."""
        e = np.exp(x - np.max(x))
        return (e / e.sum()).tolist()

## Main evaluator
class RAGEvaluator:
    """
    Automatic RAG quality evaluator.

    Scores every query across four metrics and persists results to SQLite.
    Designed to run alongside the main application on 8GB Apple Silicon
    with minimal RAM impact.

    Metric weights for composite score:
      faithfulness:       35%  (most important — catches hallucination)
      answer_relevance:   30%  (second most important — catches off-topic answers)
      context_precision:  20%  (retrieval quality signal)
      source_coverage:    15%  (proxy for grounding when NLI isn't available)
    """

    WEIGHTS = {
        "faithfulness":       0.35,
        "answer_relevance":   0.30,
        "context_precision":  0.20,
        "source_coverage":    0.15,
    }

    def __init__(
        self,
        db_path: str = DB_PATH,
        use_nli: bool = True,
        unload_nli_after_scoring: bool = True
    ):
        """
        Args:
            db_path: Path to SQLite database for storing results
            use_nli: Whether to run Tier 2 NLI faithfulness scoring
            unload_nli_after_scoring: Free NLI model RAM after each score call
        """
        self.store = EvalStore(db_path)
        self.embedding_scorer = EmbeddingScorer()
        self.faithfulness_scorer = FaithfulnessScorer()
        self.use_nli = use_nli and self.faithfulness_scorer.is_available()
        self.unload_nli = unload_nli_after_scoring

        logger.info(
            f"RAGEvaluator initialized | NLI: {'enabled' if self.use_nli else 'disabled'}"
        )

    def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        sources: list[dict],
        query_time: float = 0.0,
        save: bool = True
    ) -> EvalResult:
        """
        Score a single RAG query-response pair.

        Args:
            question:   The user's question
            answer:     The generated answer
            contexts:   Retrieved context chunk texts (list of strings)
            sources:    Source metadata dicts with 'file', 'score', 'preview' keys
            query_time: Time taken to generate the answer in seconds
            save:       Whether to persist the result to SQLite

        Returns:
            EvalResult with all metric scores
        """
        t0 = time.time()

        # Tier 1: embedding-based metrics (always runs)
        answer_relevance = self.embedding_scorer.score_answer_relevance(
            question, answer
        )
        context_precision = self.embedding_scorer.score_context_precision(
            question, contexts
        )
        source_coverage = self.embedding_scorer.score_source_coverage(
            answer, sources
        )

        # Tier 2: NLI faithfulness (runs if enabled)
        if self.use_nli:
            faithfulness, faithfulness_detail = self.faithfulness_scorer.score(
                answer, contexts
            )
            if self.unload_nli:
                self.faithfulness_scorer.unload()
            tier = "full"
        else:
            # Fall back to source_coverage as a faithfulness proxy
            faithfulness = source_coverage
            faithfulness_detail = "[]"
            tier = "basic"

        # Composite score
        composite = (
            self.WEIGHTS["faithfulness"]       * faithfulness +
            self.WEIGHTS["answer_relevance"]   * answer_relevance +
            self.WEIGHTS["context_precision"]  * context_precision +
            self.WEIGHTS["source_coverage"]    * source_coverage
        )

        # Source metadata
        scores = [s.get("score") for s in sources if s.get("score") is not None]
        avg_source_score = sum(scores) / len(scores) if scores else 0.0

        result = EvalResult(
            question=question,
            answer=answer,
            query_time=query_time,
            timestamp=datetime.now().isoformat(),
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(answer_relevance, 4),
            context_precision=round(context_precision, 4),
            source_coverage=round(source_coverage, 4),
            composite_score=round(composite, 4),
            tier=tier,
            num_sources=len(sources),
            avg_source_score=round(avg_source_score, 4),
            answer_length=len(answer),
            faithfulness_detail=faithfulness_detail,
        )

        eval_time = round(time.time() - t0, 3)
        logger.info(
            f"Eval complete | composite: {result.composite_score:.3f} | "
            f"faithfulness: {result.faithfulness:.3f} | "
            f"relevance: {result.answer_relevance:.3f} | "
            f"tier: {tier} | eval_time: {eval_time}s"
        )

        if save:
            self.store.save(result)

        return result

    def get_trend(self, days: int = 7) -> list[dict]:
        """Return daily quality trend for the last N days."""
        return self.store.get_trend(days)

    def get_summary(self) -> dict:
        """Return overall quality statistics."""
        return self.store.get_summary()

    def get_recent(self, n: int = 20) -> list[dict]:
        """Return the N most recent scored queries."""
        return self.store.get_recent(n)

    def get_low_quality(self, threshold: float = 0.5, n: int = 20) -> list[dict]:
        """Return queries that scored below threshold, for debugging."""
        return self.store.get_low_quality(threshold, n)


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences for per-claim evaluation.

    Uses a simple regex approach rather than NLTK to avoid adding
    another dependency. Works well for short factual answers.
    """
    # Split on sentence-ending punctuation followed by whitespace or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty and very short fragments (likely punctuation artifacts)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


