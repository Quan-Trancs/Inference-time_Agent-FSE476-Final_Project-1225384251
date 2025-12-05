from utils import call_model_chat_completions, MODEL
from inference_techniques import InferenceTechnique

class WorkingAgent:
    def __init__(self):
        self.technique = InferenceTechnique(self)

    def solve_and_answer(self, question):
        # Use inference technique to solve the question
        # For now, use the _call method directly until techniques are implemented
        
        answer = self.technique._call(
            question,
            temperature=0.0,
            system="You are a helpful assistant. Reply with only the final answer—no explanation."
        )
        
        return answer
