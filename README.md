# Final Project - LLM Agent with Inference Techniques

This project implements an LLM-based agent system for answering questions across multiple domains including math, coding, common sense, future prediction, and planning.

## Features added

- **Chain-of-Thought (CoT) Prompting**: Step-by-step reasoning for complex problems
- **Domain-specific evaluation**: Tracks performance by category
- **LLM-as-a-judge evaluation**: Uses self_evaluate for answer validation
- **Comprehensive reporting**: Generates statistics and wrong answer reports


## Usage

Run the main script to generate answers for the dev dataset:

```bash
python main.py
```

This will:
1. Load questions from `cse476_final_project_dev_data.json`
2. Generate answers using the agent
3. Evaluate answers using LLM-as-a-judge
4. Save results to `cse_476_final_project_answers.json`
5. Generate statistics and wrong answer reports

## Chain-of-Thought (CoT) Implementation

The project uses **Chain-of-Thought Prompting** as the primary inference technique. This approach:

- **Improves math results**: Step-by-step reasoning significantly improves accuracy on mathematical problems
- **Works across domains**: Effective for coding, planning, predictions, and common sense questions

### Known Limitation

**Token Limit Issue**: In some cases, the Chain-of-Thought approach generates responses that exceed the allowed token limit. When this happens, the final answer may be truncated or not included in the response, which can lead to incomplete or missing answers.

To mitigate this:
- The system attempts to extract the final answer from available text
- Falls back to the last meaningful line if "Final Answer:" marker is not found
- Consider adjusting `max_tokens` in the API call if this becomes a frequent issue

## Output Files

- `cse_476_final_project_answers.json`: Generated answers in the required format
- `statistics_report.txt`: Performance statistics by category
- `wrong_answers_report.txt`: Detailed list of incorrect answers with full context

## Configuration

Key parameters can be adjusted in:
- `inference_techniques.py`: `max_calls_per_question` (default: 20)
- `utils.py`: `max_tokens` in API calls (default: 128)
- `main.py`: Input/output file paths

## Evaluation

The system uses `self_evaluate` from `utils.py` which:
- Uses the LLM itself as a judge to determine correctness
- Falls back to normalized string comparison if needed
- Handles different answer formats (numeric, text, etc.)

## Performance

Current performance by domain (from statistics report):
- **MATH**: Best performance with CoT reasoning
- **COMMON_SENSE**: Moderate performance
- **CODING**: Code generation tasks
- **FUTURE_PREDICTION**: Prediction tasks
- **PLANNING**: Planning and reasoning tasks

## Notes

- The system processes questions sequentially with a progress bar
- Wrong answers are tracked with full domain, input, expected, and actual output
- All files use **UTF-8** encoding to handle special characters
