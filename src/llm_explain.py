import os
import sys
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLM_PROVIDER, LLM_MODEL, GROQ_API_KEY,
    MAX_TOKENS, TEMPERATURE, CLASS_NAMES, IDX_TO_CLASS,
)
from src.rag import retrieve

SYSTEM_PROMPT = """You are a clinical dermatology AI assistant.
You explain skin lesion classifications to patients and doctors clearly and compassionately.
Always remind users that your analysis is for educational purposes only and not a substitute for professional medical diagnosis.
Use the retrieved medical context to provide accurate, evidence-based explanations."""


def _call_groq(messages: list, system: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    full_messages = [{"role": "system", "content": system}] + messages
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=full_messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content


def _call_llm(messages: list, system: str) -> str:
    provider = LLM_PROVIDER.lower()
    if provider == "groq":
        return _call_groq(messages, system)
    raise ValueError(f"Unsupported LLM provider: {provider}")


# ── Public API ─────────────────────────────────────────────────────────────────

def explain_prediction(
    predicted_class: str,
    confidence: float,
    probabilities: dict,
) -> str:
    """Generate a clinical explanation for a prediction using RAG + LLM."""

    class_name = CLASS_NAMES[list(IDX_TO_CLASS.values()).index(predicted_class)]

    # Retrieve relevant context from DermNet knowledge base
    query   = f"{class_name} symptoms diagnosis treatment dermoscopy"
    context = retrieve(query, top_k=5)
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context
    )

    # Build top-3 probabilities string
    top3 = list(probabilities.items())[:3]
    top3_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in top3)

    user_message = f"""
A skin lesion image was analyzed by an AI classification model.

**Prediction:** {class_name} ({predicted_class})
**Confidence:** {confidence*100:.1f}%
**Top-3 probabilities:** {top3_str}

**Retrieved Medical Context from DermNet NZ:**
{context_text}

Please provide:
1. A brief description of this condition
2. Key visual/clinical features that characterize it
3. Risk level and urgency (benign vs potentially malignant)
4. Recommended next steps for the patient
5. A compassionate closing note

Keep the response clear, structured, and under 400 words.
"""

    messages = [{"role": "user", "content": user_message}]
    return _call_llm(messages, SYSTEM_PROMPT)


def answer_question(question: str, predicted_class: str = None) -> str:
    """Answer a follow-up question about the diagnosis using RAG + LLM."""

    query   = question + (f" {predicted_class}" if predicted_class else "")
    context = retrieve(query, top_k=4)
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context
    )

    user_message = f"""
The patient asks: "{question}"
{f'Context: The patient was previously diagnosed with {predicted_class}.' if predicted_class else ''}

**Retrieved Medical Context from DermNet NZ:**
{context_text}

Please answer clearly and compassionately. Remind the user to consult a dermatologist for personal medical advice.
"""

    messages = [{"role": "user", "content": user_message}]
    return _call_llm(messages, SYSTEM_PROMPT)


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("Testing RAG + Groq LLM explanation")
    print("=" * 60)

    explanation = explain_prediction(
        predicted_class="mel",
        confidence=0.8654,
        probabilities={"mel": 0.8654, "nv": 0.0821, "bkl": 0.0312, "bcc": 0.0124, "akiec": 0.0089},
    )
    print("\n[Explanation]\n")
    print(explanation)

    print("\n" + "=" * 60)
    print("Testing follow-up Q&A")
    print("=" * 60)
    answer = answer_question("Should I be worried? What should I do next?", predicted_class="mel")
    print("\n[Answer]\n")
    print(answer)
