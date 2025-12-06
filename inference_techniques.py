from collections import Counter
import re
from utils import call_model_chat_completions, MODEL

class InferenceTechnique:
    def __init__(self, agent):
        self.agent = agent
        self.call_counter = 0
        self.max_calls_per_question = 20

    def _call(self, prompt: str, temperature: float = 0.0, system: str = None) -> str:
        if self.call_counter >= self.max_calls_per_question:
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

    def reset_counter(self):
        """Reset the call counter for a new question."""
        self.call_counter = 0

    # Technique 1: Chain-of-Thought Prompting
    def chain_of_thought(self, question: str) -> str:
        """
        Chain-of-Thought Prompting: Prompts the model to think step-by-step before answering.
        Best for: All domains requiring reasoning - math, coding, planning, predictions, common sense.
        Works across: Math problems, code generation, planning tasks, future predictions, common sense questions.
        """
        # Reset counter for each new question
        self.reset_counter()
        
        chain_of_thought_prompt = f"""Solve this problem step by step. Show your reasoning clearly, then provide the final answer.

QUESTION: {question}

Think through the problem step by step:
1. Understand what is being asked
2. Break down the problem into smaller parts
3. Solve each part systematically
4. Combine the results
5. Verify your answer

Provide your reasoning, then end with:
Final Answer: <answer>"""
        
        response = self._call(
            chain_of_thought_prompt,
            temperature=0.0,
            system="You are a helpful assistant that solves problems step by step. Show your reasoning clearly."
        )
        
        # Extract final answer if present
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[-1].strip()
            # Remove any leading colons or dashes
            answer = re.sub(r'^[:\-\s]+', '', answer)
            return answer
        if "final answer:" in response.lower():
            answer = response.split("final answer:")[-1].strip()
            answer = re.sub(r'^[:\-\s]+', '', answer)
            return answer
        
        # If no explicit final answer marker, try to find the last meaningful line
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        if lines:
            # Return the last non-empty line
            return lines[-1]
        
        return response.strip() 
