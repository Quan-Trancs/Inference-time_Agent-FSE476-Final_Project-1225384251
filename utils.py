import os, json, re
import requests # type: ignore

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL    = os.getenv("MODEL_NAME", "bens_model")

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 800,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = response.status_code
        response_headers = dict(response.headers)
        if status == 200:
            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": response_headers}
        else:
            # try best-effort to surface error text
            error_text = None
            try:
                error_text = response.json()
            except Exception:
                error_text = response.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(error_text), "headers": response_headers}
    except requests.RequestException as exception:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(exception), "headers": {}}

def classify_domain(question: str, model=MODEL) -> str:
    """
    Classify the domain of a question using the LLM.
    Returns: 'math', 'coding', 'planning', 'future_prediction', or 'common_sense'
    Always returns one of the valid domains, never 'unknown'.
    """
    system = "You are a domain classifier. You MUST reply with exactly one word from: math, coding, planning, future_prediction, common_sense. Choose the best match."
    prompt = f"""Classify the domain of this question. You MUST choose exactly one from: math, coding, planning, future_prediction, common_sense.

QUESTION: {question}

Reply with only one word:"""
    
    response = call_model_chat_completions(
        prompt,
        system=system,
        model=model,
        temperature=0.0,
    )
    
    domain = (response.get("text") or "").strip().lower()
    
    # Normalize the response - always return a valid domain
    if "math" in domain:
        return "math"
    elif "coding" in domain or "code" in domain:
        return "coding"
    elif "planning" in domain:
        return "planning"
    elif "future" in domain or "prediction" in domain:
        return "future_prediction"
    elif "common" in domain or "sense" in domain:
        return "common_sense"
    else:
        # Fallback: default to common_sense if classification fails
        return "common_sense"

def self_evaluate(question, prediction, expected_answer, model=MODEL):
    """
    Use the model itself as a strict grader.
    Returns True if the model says the prediction matches the expected answer; else False.
    Falls back to a simple normalized string compare if the model's reply is malformed.
    """
    system = "You are a strict grader. Reply with exactly True or False. No punctuation. No explanation."
    prompt = f"""You are grading a question-answer pair.

Return exactly True if the PREDICTION would be accepted as correct for the EXPECTED_ANSWER.
Otherwise, return False.

QUESTION:
{question}

PREDICTION:
{prediction}

EXPECTED_ANSWER:
{expected_answer}

Answer with exactly: True or False
"""

    response = call_model_chat_completions(
        prompt,
        system=system,
        model=model,
        temperature=0.0,
    )

    reply = (response.get("text") or "").strip().lower()
    if reply.startswith("true"):
        return True
    if reply.startswith("false"):
        return False

    # Fallback: simple normalization-based equality
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    return norm(prediction) == norm(expected_answer)
