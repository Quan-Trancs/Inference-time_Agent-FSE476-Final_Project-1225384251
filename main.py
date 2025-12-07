import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc="", unit=""):
        return iterable
from agent import WorkingAgent
from utils import self_evaluate, MODEL


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="surrogatepass") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data

def check_answer(question: str, prediction: str, expected: str) -> bool:
    """Check if the answer is correct using self_evaluate (LLM-as-a-judge)."""
    # Convert to strings to handle non-string types (bool, int, etc.)
    question_str = str(question) if question is not None else ""
    prediction_str = str(prediction) if prediction is not None else ""
    expected_str = str(expected) if expected is not None else ""
    
    # Use self_evaluate to check if prediction matches expected answer
    return self_evaluate(question_str, prediction_str, expected_str, model=MODEL)

def build_answers(questions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    # Initialize the agent once for all questions
    agent = WorkingAgent()
    answers = []
    
    # Track statistics by category
    category_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "wrong_answers": []})
    
    # Use tqdm to show progress bar
    for idx, question in enumerate(tqdm(questions, desc="Processing questions", unit="question"), start=1):
        # Use agent to solve and get an answer
        question_input = question.get("input", "")
        expected_output = question.get("output", None)  # May not exist in test data
        domain = question.get("domain", None)  # May not exist in test data
        
        result = agent.solve_and_answer(question_input)
        
        # Handle dict return (from math_chain_of_thought) or string return
        if isinstance(result, dict):
            answer_text = result.get("answer", "")
            full_response = result.get("full_response", "")
        else:
            answer_text = result
            full_response = result
        
        # Clean answer: ensure it contains only the final answer, no reasoning
        # Remove any potential reasoning markers or extra text
        answer_text = str(answer_text).strip()
        
        # Remove common prefixes that might indicate reasoning
        prefixes_to_remove = ["Final Answer:", "final answer:", "Answer:", "answer:"]
        for prefix in prefixes_to_remove:
            if answer_text.lower().startswith(prefix.lower()):
                answer_text = answer_text[len(prefix):].strip()
                # Remove leading colons, dashes, or whitespace
                answer_text = re.sub(r'^[:\-\s]+', '', answer_text)
        
        # Take only the first line if multiple lines (final answer should be single line)
        if '\n' in answer_text:
            answer_text = answer_text.split('\n')[0].strip()
        
        # Ensure answer is a string and not empty
        if not answer_text:
            answer_text = ""
        
        answers.append({"output": answer_text})
        
        # Only evaluate if expected_output exists (dev data has it, test data doesn't)
        if expected_output is not None:
            # Check if answer is correct using self_evaluate (LLM-as-a-judge)
            is_correct = check_answer(question_input, answer_text, expected_output)
            
            # Use domain from question or classify if not available
            if domain is None:
                from utils import classify_domain
                domain = classify_domain(question_input)
            
            if is_correct:
                category_stats[domain]["correct"] += 1
            else:
                category_stats[domain]["wrong"] += 1
                category_stats[domain]["wrong_answers"].append({
                    "index": idx,
                    "domain": domain,
                    "input": question_input,
                    "expected": expected_output,
                    "got": answer_text,
                    "full_response": full_response
                })
    
    return answers, dict(category_stats)

def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )

def generate_answers():
    """Generate answers for test questions and save to output file."""
    # Use test data for final submission
    input_path = Path("cse_476_final_project_test_data.json")
    output_path = Path("cse_476_final_project_answers.json")
    
    print("\n=== Generating Answers ===")
    questions = load_questions(input_path)
    print(f"Loaded {len(questions)} questions from {input_path}")
    
    answers, category_stats = build_answers(questions)
    
    with output_path.open("w", encoding="utf-8", errors="surrogatepass") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)
    
    with output_path.open("r", encoding="utf-8", errors="surrogatepass") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {output_path} "
        "and validated format successfully."
    )
    
    # Print category statistics
    print("\n" + "="*60)
    print("CATEGORY STATISTICS")
    print("="*60)
    
    total_correct = 0
    total_wrong = 0
    
    for domain in sorted(category_stats.keys()):
        stats = category_stats[domain]
        correct = stats["correct"]
        wrong = stats["wrong"]
        total = correct + wrong
        accuracy = (correct / total * 100) if total > 0 else 0
        
        total_correct += correct
        total_wrong += wrong
        
        print(f"\n{domain.upper()}:")
        print(f"  Correct: {correct}/{total} ({accuracy:.1f}%)")
        print(f"  Wrong: {wrong}/{total}")
    
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    total_questions = total_correct + total_wrong
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    print(f"Total Correct: {total_correct}/{total_questions} ({overall_accuracy:.1f}%)")
    print(f"Total Wrong: {total_wrong}/{total_questions}")
    
    # Write wrong answers by category to file
    wrong_answers_path = Path("wrong_answers_report.txt")
    with wrong_answers_path.open("w", encoding="utf-8", errors="surrogatepass") as fp:
        fp.write("="*60 + "\n")
        fp.write("WRONG ANSWERS BY CATEGORY\n")
        fp.write("="*60 + "\n")
        
        for domain in sorted(category_stats.keys()):
            wrong_answers = category_stats[domain]["wrong_answers"]
            if wrong_answers:
                fp.write(f"\n{domain.upper()} ({len(wrong_answers)} wrong):\n")
                for wrong in wrong_answers:
                    fp.write(f"\n  Question #{wrong['index']}:\n")
                    fp.write(f"    Domain: {wrong['domain']}\n")
                    fp.write(f"    Input: {wrong['input']}\n")
                    fp.write(f"    Expected: {wrong['expected']}\n")
                    fp.write(f"    Final Answer: {wrong['got']}\n")
                    if 'full_response' in wrong and wrong['full_response']:
                        fp.write(f"    Full Response:\n{wrong['full_response']}\n")
    
    if total_wrong > 0:
        print(f"\nWrong answers written to {wrong_answers_path}")

def main():
    print("\n=== Generating Answers for Dev Data ===")
    generate_answers()

if __name__ == "__main__":
    main()