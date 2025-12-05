import os, json, re
import requests

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
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

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

    r = call_model_chat_completions(
        prompt,
        system=system,
        model=model,
        temperature=0.0,
    )

    reply = (r.get("text") or "").strip().lower()
    if reply.startswith("true"):
        return True
    if reply.startswith("false"):
        return False

    # Fallback: simple normalization-based equality
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    return norm(prediction) == norm(expected_answer)
