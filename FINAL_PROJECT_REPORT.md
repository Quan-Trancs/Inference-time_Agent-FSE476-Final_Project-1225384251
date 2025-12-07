# Final Project Report: LLM Agent with Domain-Specific Inference Techniques

## Overview

This project implements an intelligent LLM-based agent system that automatically classifies question domains and applies specialized inference techniques to generate accurate answers. The system processes questions across multiple domains including mathematics, coding, planning, future prediction, and common sense reasoning.

## Agent Architecture

### High-Level Flow

1. **Domain Classification**: The agent first classifies each question's domain using an LLM-based classifier
2. **Technique Selection**: Based on the classified domain, the agent routes to the appropriate inference technique
3. **Answer Generation**: The selected technique generates an answer with step-by-step reasoning
4. **Answer Extraction**: The final answer is extracted and cleaned to ensure format compliance
5. **Evaluation**: Answers are evaluated using LLM-as-a-judge methodology

### Key Components

#### 1. Domain Classification (`utils.py`)

**Function**: `classify_domain(question: str) -> str`

**Location**: `c:\ASU\FSE 476\final_project\utils.py`, lines 51-87

**Implementation Details**:
- Uses LLM to classify questions into one of five domains: math, coding, planning, future_prediction, or common_sense
- Always returns a valid domain (never "unknown") with fallback to "common_sense"
- Temperature set to 0.0 for consistent classification

**Code Block**:
```python
def classify_domain(question: str, model=MODEL) -> str:
    system = "You are a domain classifier. You MUST reply with exactly one word from: math, coding, planning, future_prediction, common_sense. Choose the best match."
    prompt = f"""Classify the domain of this question. You MUST choose exactly one from: math, coding, planning, future_prediction, common_sense.
    
    QUESTION: {question}
    
    Reply with only one word:"""
    
    response = call_model_chat_completions(prompt, system=system, model=model, temperature=0.0)
    # ... normalization logic ...
```

#### 2. Agent Router (`agent.py`)

**Class**: `WorkingAgent`

**Location**: `c:\ASU\FSE 476\final_project\agent.py`, lines 4-29

**Implementation Details**:
- Routes questions to domain-specific techniques based on classification
- Math questions → `math_chain_of_thought` (4-phase structured reasoning)
- Future prediction questions → `self_consistency` (7-sample majority voting)
- All other domains → `chain_of_thought` (step-by-step reasoning)

**Code Block**:
```python
def solve_and_answer(self, question):
    domain = classify_domain(question)
    
    if domain == "math":
        result = self.technique.math_chain_of_thought(question)
    elif domain == "future_prediction":
        answer = self.technique.self_consistency(question, samples=7)
        return {"answer": answer, "full_response": answer}
    else:
        result = self.technique.chain_of_thought(question)
    
    return result
```

#### 3. Inference Techniques (`inference_techniques.py`)

**Class**: `InferenceTechnique`

**Location**: `c:\ASU\FSE 476\final_project\inference_techniques.py`

##### Technique 1: Math Chain-of-Thought with 4-Phase Process

**Function**: `math_chain_of_thought(question: str) -> dict`

**Location**: Lines 78-146

**Implementation Details**:
- Implements a structured 4-phase approach: SETUP, PLAN, EXECUTION, VERIFICATION
- SETUP: Lists variables and constraints
- PLAN: Identifies formulas, theorems, and solution strategies
- EXECUTION: Shows numbered steps with concise math
- VERIFICATION: Plugs answer back into constraints to verify correctness
- Returns both extracted final answer and full response for debugging
- Uses temperature 0.1 for consistency

**Key Code Block**:
```python
meta_prompt = f"""You are an expert mathematician. Solve the following problem using a strict 4-phase process.

QUESTION: {question}

INSTRUCTIONS:
1. PHASE 1: SETUP - List every variable given in the text. List every constraint. Define what exact value is requested.
2. PHASE 2: PLAN - State the formula or theorem you will use. Look for patterns or simplification techniques.
3. PHASE 3: EXECUTION - Show your steps numbered 1, 2, 3... Keep steps concise but show the math.
4. PHASE 4: VERIFICATION - Take your final candidate answer and Plug It Back into the original constraints.

FORMAT:
[SETUP]...
[PLAN]...
[EXECUTION]
1. ...
2. ...
[VERIFICATION]
...
final answer: <numeric_answer_only>

SOLVE NOW:"""
```

##### Technique 2: Self-Consistency (for Future Predictions)

**Function**: `self_consistency(question: str, samples: int = 7) -> str`

**Location**: Lines 148-200

**Implementation Details**:
- Generates 7 independent answers with temperature 0.7 for diversity
- Uses `answer{ANSWER}` format for consistent extraction
- Applies majority voting to select the most common answer
- Reduces errors through aggregation

**Key Code Block**:
```python
for _ in range(samples):
    response = self._call(
        f"""{question}
        
        IMPORTANT:
        Your final answer MUST end with this exact format:
        answer{{YOUR_ANSWER}}
        Do not add anything else.""",
        temperature=0.7
    )
    # Extract answer from answer{ANSWER} format
    predictions.append(answer)

count = Counter(predictions)
most_common = count.most_common(1)[0][0]
return most_common
```

##### Technique 3: Chain-of-Thought (General Purpose)

**Function**: `chain_of_thought(question: str) -> dict`

**Location**: Lines 29-76

**Implementation Details**:
- Prompts model to think step-by-step before answering
- Extracts final answer from "Final Answer:" markers
- Returns both answer and full response
- Works across all non-math domains

#### 4. Answer Processing (`main.py`)

**Function**: `build_answers(questions)`

**Location**: `c:\ASU\FSE 476\final_project\main.py`, lines 32-95

**Implementation Details**:
- Processes questions sequentially with progress bar
- Cleans answers to ensure only final answer (no reasoning) in output
- Removes prefixes like "Final Answer:", takes first line only
- Validates format: string type, < 5000 characters, "output" field present

**Key Code Block**:
```python
# Clean answer: ensure it contains only the final answer, no reasoning
answer_text = str(answer_text).strip()

# Remove common prefixes that might indicate reasoning
prefixes_to_remove = ["Final Answer:", "final answer:", "Answer:", "answer:"]
for prefix in prefixes_to_remove:
    if answer_text.lower().startswith(prefix.lower()):
        answer_text = answer_text[len(prefix):].strip()
        answer_text = re.sub(r'^[:\-\s]+', '', answer_text)

# Take only the first line if multiple lines
if '\n' in answer_text:
    answer_text = answer_text.split('\n')[0].strip()

answers.append({"output": answer_text})
```

#### 5. Evaluation System (`utils.py`)

**Function**: `self_evaluate(question, prediction, expected_answer)`

**Location**: `c:\ASU\FSE 476\final_project\utils.py`, lines 91-128

**Implementation Details**:
- Uses LLM-as-a-judge to evaluate answer correctness
- Falls back to normalized string comparison if LLM response is malformed
- Handles different answer formats (numeric, text, etc.)

## Key Design Decisions

### 1. Domain-Specific Techniques

**Rationale**: Different question types require different reasoning approaches:
- Math problems benefit from structured 4-phase verification
- Future predictions benefit from majority voting across multiple samples
- General questions benefit from step-by-step chain-of-thought

**Implementation**: Automatic domain classification routes to appropriate technique

### 2. 4-Phase Math Process

**Rationale**: Mathematical problems often have multiple solution paths. The 4-phase approach ensures:
- All constraints are identified upfront (SETUP)
- Strategy is planned before execution (PLAN)
- Verification catches errors before submission (VERIFICATION)

**Result**: Improved accuracy on math problems by reducing calculation errors

### 3. Self-Consistency for Predictions

**Rationale**: Future prediction questions have inherent uncertainty. Generating multiple answers and selecting the most common reduces variance and improves reliability.

**Implementation**: 7 samples with temperature 0.7, majority voting

### 4. Token Limit Increase

**Rationale**: Chain-of-thought reasoning requires more tokens. Increased from 128 to 800 to:
- Accommodate full reasoning chains
- Include verification steps
- Prevent final answer truncation

**Location**: `utils.py`, line 29: `"max_tokens": 800`

### 5. Answer Cleaning

**Rationale**: Auto-grader requires only final answers, no reasoning. Cleaning ensures:
- No intermediate steps in output
- No "Final Answer:" prefixes
- Single-line answers only

**Implementation**: `main.py`, lines 58-73

## File Structure

```
final_project/
├── agent.py                  # WorkingAgent class with domain routing
├── inference_techniques.py   # Three inference techniques implementation
├── utils.py                  # API utilities, domain classification, evaluation
├── main.py                   # Main execution script with answer processing
├── cse_476_final_project_test_data.json    # Test dataset (input only)
├── cse476_final_project_dev_data.json     # Dev dataset (with expected answers)
└── README.md                 # Detailed setup and usage instructions
```

## How to Reproduce Results

### Prerequisites

1. Python 3.x
2. Required packages: `requests`, `tqdm` (optional, for progress bar)
3. Access to the LLM API endpoint (configured via environment variables)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Quan-Trancs/Inference-time_Agent-FSE476-Final_Project-1225384251
cd Inference-time_Agent-FSE476-Final_Project-1225384251
```

2. Install dependencies:
```bash
pip install requests tqdm
```

3. Configure environment variables (optional, defaults provided):
```bash
export OPENAI_API_KEY="cse476"
export API_BASE="http://10.4.58.53:41701/v1"
export MODEL_NAME="bens_model"
```

### Running the Agent

**For test data (submission format)**:
```bash
python main.py
```

This will:
1. Load questions from `cse_476_final_project_test_data.json`
2. Classify each question's domain
3. Apply appropriate inference technique
4. Generate answers in format: `[{"output": "answer1"}, {"output": "answer2"}, ...]`
5. Save to `cse_476_final_project_answers.json`

**For dev data (with evaluation)**:
Modify `main.py` line 99 to use `cse476_final_project_dev_data.json` instead.

### Output Format

The generated JSON file follows the exact format required by the auto-grader:
- Each entry has exactly one field: `"output"`
- The value is a string containing only the final answer
- No reasoning, intermediate steps, or tool traces included
- Validated to be < 5000 characters per answer

## Performance Characteristics

### Token Usage

- **Domain Classification**: 1 API call per question (~50-100 tokens)
- **Math Questions**: 1 API call (~800 tokens max)
- **Future Predictions**: 7 API calls per question (~100 tokens each)
- **Other Domains**: 1 API call (~800 tokens max)

**Total**: Approximately 1-8 API calls per question depending on domain

### Known Limitations

1. **Token Limit**: While increased to 800, very complex problems may still truncate
   - Mitigation: Answer extraction handles truncated responses
   - Full response stored for debugging

2. **Domain Classification**: May occasionally misclassify edge cases
   - Mitigation: Fallback to common_sense ensures valid technique is always used

3. **Answer Extraction**: If model doesn't follow format exactly, extraction may fail
   - Mitigation: Multiple fallback strategies (marker detection, last line extraction)

## GitHub Repository

**Repository Link**: https://github.com/Quan-Trancs/Inference-time_Agent-FSE476-Final_Project-1225384251

The repository includes:
- Complete source code with detailed comments
- README.md with setup instructions
- All necessary configuration files
- Example outputs and reports

## Conclusion

This agent system demonstrates effective use of domain-specific inference techniques to improve answer quality across diverse question types. The 4-phase math process significantly improves mathematical problem-solving accuracy, while self-consistency provides robust predictions for uncertain questions. The automatic domain classification ensures optimal technique selection without manual intervention.
