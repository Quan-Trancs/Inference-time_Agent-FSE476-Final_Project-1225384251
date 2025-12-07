# Final Project - LLM Agent with Domain-Specific Inference Techniques

This project implements an intelligent LLM-based agent system that automatically classifies question domains and applies specialized inference techniques to generate accurate answers across multiple domains: mathematics, coding, planning, future prediction, and common sense reasoning.

## Features

- **Automatic Domain Classification**: LLM-based classifier routes questions to appropriate techniques
- **Domain-Specific Techniques**:
  - **Math**: 4-phase structured reasoning (SETUP, PLAN, EXECUTION, VERIFICATION)
  - **Future Prediction**: Self-consistency with 7-sample majority voting
  - **Other Domains**: Chain-of-Thought step-by-step reasoning
- **LLM-as-a-judge evaluation**: Uses self_evaluate for answer validation
- **Comprehensive reporting**: Generates statistics and wrong answer reports with full responses
- **Format compliance**: Ensures submission-ready output format

## Installation

### Prerequisites

- Python 3.x
- Required packages: `requests`, `tqdm` (optional, for progress bar)

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

## Usage

### For Test Data (Final Submission)

Run the main script to generate answers for the test dataset:

```bash
python main.py
```

This will:
1. Load questions from `cse_476_final_project_test_data.json`
2. Classify each question's domain
3. Apply appropriate inference technique based on domain
4. Generate answers in the required format
5. Save results to `cse_476_final_project_answers.json`
6. Validate format compliance

### For Dev Data (With Evaluation)

To test with dev data (includes expected answers for evaluation), modify `main.py` line 99:
- Change `cse_476_final_project_test_data.json` to `cse476_final_project_dev_data.json`

This will additionally:
- Evaluate answers using LLM-as-a-judge
- Generate statistics by domain
- Create wrong answers report with full responses

## Architecture Overview

### Agent Flow

1. **Domain Classification** (`utils.py::classify_domain`)
   - Classifies question into: math, coding, planning, future_prediction, or common_sense
   - Uses LLM with temperature 0.0 for consistent classification
   - Always returns a valid domain (fallback to common_sense if unclear)

2. **Technique Routing** (`agent.py::solve_and_answer`)
   - Math → `math_chain_of_thought()` (4-phase structured process)
   - Future Prediction → `self_consistency()` (7 samples, majority voting)
   - Other → `chain_of_thought()` (step-by-step reasoning)

3. **Answer Generation** (`inference_techniques.py`)
   - Each technique generates reasoning and extracts final answer
   - Returns both answer and full response for debugging

4. **Answer Cleaning** (`main.py::build_answers`)
   - Removes reasoning markers ("Final Answer:", etc.)
   - Takes only first line (ensures single-line answers)
   - Validates format compliance

## Implementation Details

### Math Chain-of-Thought (4-Phase Process)

**File**: `inference_techniques.py`, function `math_chain_of_thought()`, lines 78-146

**Process**:
1. **SETUP**: Lists variables, constraints, and what's requested
2. **PLAN**: Identifies formulas/theorems and solution strategy
3. **EXECUTION**: Shows numbered steps with concise math
4. **VERIFICATION**: Plugs answer back into constraints

**Key Features**:
- Temperature 0.1 for consistency
- Returns dict with `answer` and `full_response`
- Extracts final answer from "final answer:" marker

### Self-Consistency (Future Predictions)

**File**: `inference_techniques.py`, function `self_consistency()`, lines 148-200

**Process**:
- Generates 7 independent answers with temperature 0.7
- Uses `answer{ANSWER}` format for extraction
- Applies majority voting via Counter

**Key Features**:
- 7 samples for robust aggregation
- Temperature 0.7 for diversity
- Consistent format extraction

### Chain-of-Thought (General)

**File**: `inference_techniques.py`, function `chain_of_thought()`, lines 29-76

**Process**:
- Prompts step-by-step reasoning
- Extracts final answer from markers
- Returns both answer and full response

### Domain Classification

**File**: `utils.py`, function `classify_domain()`, lines 51-87

**Process**:
- Single LLM call to classify domain
- Normalizes response to valid domain
- Fallback to common_sense if classification unclear

### Answer Format Compliance

**File**: `main.py`, function `build_answers()`, lines 58-73

**Ensures**:
- Only final answer in output (no reasoning)
- Single-line answers
- Removes all prefixes and markers
- Validates string type and length (< 5000 chars)

## Key Design Decisions

### 1. Domain-Specific Techniques

Different question types require different reasoning approaches:
- **Math**: Structured 4-phase process with verification reduces calculation errors
- **Future Prediction**: Self-consistency with majority voting handles uncertainty
- **General**: Chain-of-thought provides step-by-step reasoning

### 2. Token Limit: 128 → 800

**Rationale**: Chain-of-thought reasoning requires more tokens. Increased to:
- Accommodate full reasoning chains
- Include verification steps for math
- Prevent final answer truncation

**Location**: `utils.py`, line 29

### 3. Answer Cleaning

**Rationale**: Auto-grader requires only final answers. Cleaning ensures:
- No intermediate steps in output
- No "Final Answer:" prefixes
- Single-line answers only

**Location**: `main.py`, lines 58-73

## Output Files

- `cse_476_final_project_answers.json`: Generated answers in the required format
- `statistics_report.txt`: Performance statistics by category
- `wrong_answers_report.txt`: Detailed list of incorrect answers with full context

## File Structure

```
final_project/
├── agent.py                  # WorkingAgent class with domain routing
├── inference_techniques.py   # Three inference techniques implementation
├── utils.py                  # API utilities, domain classification, evaluation
├── main.py                   # Main execution script with answer processing
├── cse_476_final_project_test_data.json    # Test dataset (input only)
├── cse476_final_project_dev_data.json     # Dev dataset (with expected answers)
├── README.md                 # This file
└── FINAL_PROJECT_REPORT.md   # Detailed project report
```

## Configuration

Key parameters can be adjusted in:
- `inference_techniques.py`: `max_calls_per_question` (default: 20)
- `utils.py`: `max_tokens` in API calls (default: 800)
- `main.py`: Input/output file paths (line 99-100)

## Output Format

The generated JSON file follows the exact format required by the auto-grader:
- Each entry: `{"output": "final_answer_string"}`
- Output contains only the final answer (no reasoning)
- Validated: string type, < 5000 characters, "output" field present

## Evaluation

The system uses `self_evaluate` from `utils.py` (lines 91-128) which:
- Uses the LLM itself as a judge to determine correctness
- Falls back to normalized string comparison if needed
- Handles different answer formats (numeric, text, etc.)

**Note**: Evaluation only runs on dev data (which has expected answers). Test data does not include expected answers.

## Performance Characteristics

### API Call Usage

- **Domain Classification**: 1 call per question (~50-100 tokens)
- **Math Questions**: 1 call (~800 tokens max)
- **Future Predictions**: 7 calls per question (~100 tokens each)
- **Other Domains**: 1 call (~800 tokens max)

**Total**: 1-8 API calls per question depending on domain

### Known Limitations

1. **Token Limit**: While increased to 800, very complex problems may still truncate
   - Mitigation: Answer extraction handles truncated responses, full response stored for debugging

2. **Domain Classification**: May occasionally misclassify edge cases
   - Mitigation: Fallback to common_sense ensures valid technique is always used

3. **Answer Extraction**: If model doesn't follow format exactly, extraction may fail
   - Mitigation: Multiple fallback strategies (marker detection, last line extraction)

## Troubleshooting

### Issue: "ERROR: max call limit reached"
- **Cause**: Call counter not resetting between questions
- **Solution**: Already fixed - counter resets at start of each technique

### Issue: Answers contain reasoning instead of just final answer
- **Cause**: Answer cleaning not working properly
- **Solution**: Check `main.py` lines 58-73 for cleaning logic

### Issue: Domain misclassification
- **Cause**: LLM classification uncertainty
- **Solution**: System automatically falls back to common_sense domain

## GitHub Repository

**Repository Link**: https://github.com/Quan-Trancs/Inference-time_Agent-FSE476-Final_Project-1225384251

The repository includes:
- Complete source code with detailed comments
- This README with setup instructions
- FINAL_PROJECT_REPORT.md with detailed implementation description
- All necessary configuration files
