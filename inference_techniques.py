from utils import call_model_chat_completions, MODEL

class InferenceTechnique:
    def __init__(self, agent):
        self.agent = agent
        self.call_counter = 0
        self.max_calls = 20

    def _call(self, prompt: str, temperature: float = 0.0, system: str = None) -> str:
        if self.call_counter >= self.max_calls:
            return "ERROR: max call limit reached"
        self.call_counter += 1
        response = call_model_chat_completions(
            prompt,
            system=system or "You are a helpful assistant. Reply with only the final answer—no explanation.",
            model=MODEL,
            temperature=temperature,
        )
        if not response.get("ok"):
            return f"ERROR status={response.get('status')} {response.get('error')}"
        return (response.get("text") or "").strip()

    # Technique 1: 


    # Technique 2:


    # Technique 3:
