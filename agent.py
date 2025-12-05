from utils import call_model_chat_completions, MODEL

class WorkingAgent:
    def __init__(self):
        pass

    def solve_and_answer(self, question):
        # Simple API call to get the answer
        response = call_model_chat_completions(
            prompt=question,
            system="You are a helpful assistant. Reply with only the final answer—no explanation.",
            model=MODEL,
            temperature=0.0,
        )
        
        # Extract the answer text from the response
        if response.get("ok"):
            return response.get("text", "").strip()
        else:
            return f"Error: {response.get('error', 'Unknown error')}"
