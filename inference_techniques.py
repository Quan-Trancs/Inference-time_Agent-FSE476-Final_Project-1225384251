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
    def chain_of_thought(self, question: str) -> dict:
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
        final_answer = None
        if "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            # Remove any leading colons or dashes
            final_answer = re.sub(r'^[:\-\s]+', '', final_answer)
        elif "final answer:" in response.lower():
            final_answer = response.split("final answer:")[-1].strip()
            final_answer = re.sub(r'^[:\-\s]+', '', final_answer)
        else:
            # If no explicit final answer marker, try to find the last meaningful line
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            if lines:
                final_answer = lines[-1]
            else:
                final_answer = response.strip()
        
        # Return both final answer and full response
        return {
            "answer": final_answer,
            "full_response": response.strip()
        }

    # Technique 1 + 2 (variant): Math Chain-of-Thought with Meta Prompting
    def math_chain_of_thought(self, question: str) -> str:
        """
        Math Chain-of-Thought with Meta Prompting: Uses meta-prompting for concise step-by-step math solutions.
        Best for: Mathematical problems requiring strategic planning and step-by-step reasoning.
        Works by: Prompting the model to solve concisely with numbered steps and minimal token usage.
        """
        # Reset counter for each new question
        self.reset_counter()

        meta_prompt = f"""You are an expert mathematician. Solve the following problem using a strict 4-phase process.

QUESTION: {question}

INSTRUCTIONS:
1. PHASE 1: SETUP
   - List every variable given in the text.
   - List every constraint (e.g., "x must be integer", "x > 0").
   - Define what exact value is requested.

2. PHASE 2: PLAN
   - State the formula or theorem you will use.
   - If the numbers are large, look for a pattern or simplification technique.
   - If it is a counting problem, explicitly state the method (e.g., "Complementary Counting", "Stars and Bars").

3. PHASE 3: EXECUTION
   - Show your steps numbered 1, 2, 3...
   - Keep steps concise but show the math.

4. PHASE 4: VERIFICATION (Crucial)
   - Take your final candidate answer and Plug It Back into the original constraints.
   - If the check fails, BACKTRACK and re-solve.
   - If the check passes, write the final answer.

FORMAT:
[SETUP]
...
[PLAN]
...
[EXECUTION]
1. ...
2. ...
[VERIFICATION]
...
final answer: <numeric_answer_only>

SOLVE NOW:"""
        
        response = self._call(
            meta_prompt,
            temperature=0.1,
            system="You are an expert mathematician who solves problems step by step concisely."
        )
        
        # Extract final answer if present
        final_answer = None
        if "final answer:" in response.lower():
            final_answer = response.split("final answer:")[-1].strip()
            # Remove any leading colons, dashes, or whitespace
            final_answer = re.sub(r'^[:\-\s]+', '', final_answer)
        else:
            # If no explicit final answer marker, use the full response as answer
            final_answer = response.strip()
        
        # Return both final answer and full response
        return {
            "answer": final_answer,
            "full_response": response.strip()
        }

    # Technique 2: Self-Consistency (General for all domains except math)
    def self_consistency(self, question: str, samples: int = 7) -> str:
        """
        Self-Consistency: Generates multiple answers and selects the most consistent one.
        """
        # Reset counter for each new question
        self.reset_counter()
        
        predictions = []
        
        for _ in range(samples):
            response = self._call(
                f"""
                    {question}

                    IMPORTANT:

                    Your final answer MUST end with this exact format:

                    answer{{YOUR_ANSWER}}

                    Do not add anything else.

                    """,
                temperature=0.7
            )
            
            answer = response.strip()
            
            # Extract answer from answer{ANSWER} format
            if "answer{" in answer.lower():
                # Find the start of answer{ and extract content
                start_idx = answer.lower().find("answer{")
                if start_idx != -1:
                    # Remove everything before answer{
                    answer = answer[start_idx:]
                    # Extract content between braces
                    if "{" in answer and "}" in answer:
                        start_brace = answer.find("{") + 1
                        end_brace = answer.find("}")
                        answer = answer[start_brace:end_brace].strip()
                    else:
                        # No closing brace, take everything after answer{
                        answer = answer[answer.lower().find("answer{") + 7:].strip()
            
            predictions.append(answer)
        
        # Count frequencies
        count = Counter(predictions)
        most_common = count.most_common(1)[0][0]
        
        return most_common 
