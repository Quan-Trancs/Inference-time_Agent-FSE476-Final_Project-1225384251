from utils import call_model_chat_completions, MODEL
from inference_techniques import InferenceTechnique

class WorkingAgent:
    def __init__(self):
        self.technique = InferenceTechnique(self)

    def solve_and_answer(self, question):
        # Use Chain-of-Thought technique for step-by-step reasoning
        answer = self.technique.chain_of_thought(question)
        return answer
