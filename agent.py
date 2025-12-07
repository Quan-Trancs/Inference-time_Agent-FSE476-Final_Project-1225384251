from utils import call_model_chat_completions, MODEL
from inference_techniques import InferenceTechnique

class WorkingAgent:
    def __init__(self):
        self.technique = InferenceTechnique(self)

    def solve_and_answer(self, question):
        # Use Math Chain-of-Thought technique for math problems
        result = self.technique.math_chain_of_thought(question)
        # Return dict with both answer and full_response
        return result
