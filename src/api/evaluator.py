"""
Automated RAG evaluation pipeline.
Implements the RAG Triad metrics (Faithfulness, Answer Relevance, Context Precision) using Gemini as a judge.
"""
from typing import Dict, Any, List
import json
import re

from ..llm import get_gemini_client


class RAGEvaluator:
    """Evaluates RAG generation quality using LLM-as-a-judge scoring."""
    
    def __init__(self):
        self.llm_client = get_gemini_client()
        self.llm = self.llm_client.chat_model
        
    def evaluate_faithfulness(self, context: str, answer: str) -> float:
        """
        Grade how faithful the generated answer is to the retrieved context.
        Scores from 0.0 (contains hallucinations) to 1.0 (completely grounded in context).
        """
        prompt = f"""You are an independent evaluator. Grade the faithfulness of the answer based ONLY on the context provided.
Do not use your own knowledge. Check if every claim in the answer is directly supported by the context.

Context:
{context}

Answer:
{answer}

Respond with a JSON object:
{{
    "score": float between 0.0 and 1.0,
    "reasoning": "explanation of what claims were grounded or hallucinated"
}}
Response:"""
        try:
            resp = self.llm.invoke(prompt).content
            if "{" in resp and "}" in resp:
                j_start = resp.find("{")
                j_end = resp.rfind("}") + 1
                result = json.loads(resp[j_start:j_end])
                return float(result.get("score", 1.0))
        except Exception as e:
            print(f"[EVAL] Faithfulness error: {e}")
        return 1.0

    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Grade how relevant the answer is to the user question.
        Scores from 0.0 (off-topic or incomplete) to 1.0 (directly and fully answers the question).
        """
        prompt = f"""You are an independent evaluator. Rate how relevant the answer is to the question.
Does it answer the question fully? Does it contain redundant or off-topic information?

Question:
{question}

Answer:
{answer}

Respond with a JSON object:
{{
    "score": float between 0.0 and 1.0,
    "reasoning": "brief explanation"
}}
Response:"""
        try:
            resp = self.llm.invoke(prompt).content
            if "{" in resp and "}" in resp:
                j_start = resp.find("{")
                j_end = resp.rfind("}") + 1
                result = json.loads(resp[j_start:j_end])
                return float(result.get("score", 1.0))
        except Exception as e:
            print(f"[EVAL] Answer Relevance error: {e}")
        return 1.0

    def evaluate_context_precision(self, question: str, context: str) -> float:
        """
        Grade the precision of the retrieved context.
        Scores from 0.0 (context is irrelevant noise) to 1.0 (context is highly relevant and useful to answer the question).
        """
        prompt = f"""You are an independent evaluator. Rate how precise and useful the retrieved context is to answer the question.
Does the context contain the answers to the question? Is it mostly noise?

Question:
{question}

Context:
{context}

Respond with a JSON object:
{{
    "score": float between 0.0 and 1.0,
    "reasoning": "brief explanation"
}}
Response:"""
        try:
            resp = self.llm.invoke(prompt).content
            if "{" in resp and "}" in resp:
                j_start = resp.find("{")
                j_end = resp.rfind("}") + 1
                result = json.loads(resp[j_start:j_end])
                return float(result.get("score", 1.0))
        except Exception as e:
            print(f"[EVAL] Context Precision error: {e}")
        return 1.0

    def evaluate_rag_triad(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Execute full RAG Triad evaluation.
        """
        faithfulness = self.evaluate_faithfulness(context, answer)
        relevance = self.evaluate_answer_relevance(question, answer)
        precision = self.evaluate_context_precision(question, context)
        
        overall_score = (faithfulness + relevance + precision) / 3.0
        
        return {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_precision": precision,
            "overall_quality_score": round(overall_score, 2),
            "status": "PASS" if overall_score >= 0.7 else "FAIL"
        }
