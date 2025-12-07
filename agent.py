from utils import call_model_chat_completions, MODEL, classify_domain
from inference_techniques import InferenceTechnique

class WorkingAgent:
    def __init__(self):
        self.technique = InferenceTechnique(self)

    def solve_and_answer(self, question):
        # Classify the domain of the question
        domain = classify_domain(question)
        
        # Use appropriate technique based on domain
        if domain == "math":
            result = self.technique.math_chain_of_thought(question)
            # Extract final answer from dict if it's a dict
            if isinstance(result, dict):
                return result
            return {"answer": result, "full_response": result}
        elif domain == "future_prediction":
            answer = self.technique.self_consistency(question, samples=7)
            return {"answer": answer, "full_response": answer}
        else:
            # Use regular chain-of-thought for other domains
            result = self.technique.chain_of_thought(question)
            # chain_of_thought now returns dict with answer and full_response
            if isinstance(result, dict):
                return result
            return {"answer": result, "full_response": result}
