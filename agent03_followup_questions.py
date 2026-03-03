"""
Agent 03: Follow-Up Question Agent

Purpose:
Selects the next safe, predefined follow-up question
based on similarity level and previous responses.

Design Philosophy:
- No LLM generation (reduces hallucination risk)
- Uses controlled template bank
- Prevents repeated questions
"""

from text_templates import FOLLOWUP_QUESTIONS


def select_next_question(similarity: str, previous_answers: dict):
    """
    Selects next question safely from predefined template list.
    """

    VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}

    # Normalize similarity input
    similarity_clean = similarity.strip().upper()

    # Safety fallback to LOW if unexpected input
    if similarity_clean not in VALID_LEVELS:
        similarity_clean = "LOW"

    # Retrieve candidate questions for similarity tier
    candidate_questions = FOLLOWUP_QUESTIONS[similarity_clean]

    # Remove already asked questions to avoid repetition
    remaining = [
        q for q in candidate_questions
        if q not in previous_answers.keys()
    ]

    # If no remaining questions, return None
    if not remaining:
        return None

    # Return next question in deterministic order
    return remaining[0]