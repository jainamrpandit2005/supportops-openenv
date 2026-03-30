can you provide hugging face readme.md just edit that version to this # Email Triage OpenEnv

A real-world OpenEnv environment for training and evaluating AI agents to perform email triage and prioritization tasks.

## Environment Description

The Email Triage environment simulates a realistic email management scenario where an AI agent must read emails, categorize them, and prioritize responses. This is a genuine task that email users and customer support teams perform daily.

### Motivation

Email management is a crucial skill for productivity. By creating an environment where agents learn to triage emails effectively, we can evaluate their understanding of natural language, prioritization logic, and task planning.

## Action and Observation Spaces

### Observation Space

```python
class EmailObservation(BaseModel):
    inbox: List[Email]
    current_email_id: Optional[str]
    current_email_content: Optional[str]
    email_count: int
    processed_count: int
    inbox_status: str
```

### Action Space

```python
class EmailAction(BaseModel):
    action_type: str  # "read", "categorize", "mark_spam", "mark_urgent", "archive", "move_folder"
    email_id: Optional[str]
    category: Optional[str]  # "work", "personal", "spam", "urgent"
    folder: Optional[str]
```

## Task Descriptions

### Task 1: Basic Email Categorization (Easy)
**Objective:** Categorize 10 emails correctly into work, personal, spam, or urgent.
**Expected Difficulty:** Easy
**Grader:** Checks if agent categorizes emails accurately based on content analysis.
**Success Criteria:** Score ≥ 0.8 (80% accurate categorization)

### Task 2: Email Prioritization with Actions (Medium)
**Objective:** Triage 20 emails, categorize, and take appropriate actions (archive, mark urgent).
**Expected Difficulty:** Medium
**Grader:** Evaluates both categorization accuracy and appropriate action selection.
**Success Criteria:** Score ≥ 0.7 (70% accuracy on categorization + action correctness)

### Task 3: Complex Inbox Management (Hard)
**Objective:** Manage 50 emails with multiple senders, handle patterns, and filter spam intelligently.
**Expected Difficulty:** Hard
**Grader:** Checks pattern recognition, spam detection accuracy, and overall inbox cleanliness.
**Success Criteria:** Score ≥ 0.6 (comprehensive inbox management with <10% false positives)

## Setup Instructions

### Prerequisites

- Docker
- Python 3.11+
- Hugging Face account (for HF_TOKEN)

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd email-triage-openenv

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4-turbo"
export HF_TOKEN="your-hugging-face-token"
export OPENAI_API_KEY="your-openai-key"

# Run inference
python inference.py
```

### Docker Deployment

```bash
# Build the image
docker build -t email-triage-env:latest .

# Run the container
docker run -e API_BASE_URL="https://api.openai.com/v1" \
           -e MODEL_NAME="gpt-4-turbo" \
           -e HF_TOKEN="your-token" \
           -e OPENAI_API_KEY="your-key" \
           email-triage-env:latest
```

### Hugging Face Space Deployment

1. Create a new Space on Hugging Face
2. Push this repository to the Space
3. The Dockerfile will automatically build and deploy
4. Add environment secrets in Space settings

## Baseline Scores

Baseline inference script using GPT-4-turbo:

- **Task 1 (Easy):** 0.88
- **Task 2 (Medium):** 0.75
- **Task 3 (Hard):** 0.62

Run `python inference.py` to reproduce these scores.

## Project Structure

```
email-triage-openenv/
├── README.md
├── requirements.txt
├── Dockerfile
├── openenv.yaml
├── inference.py
├── env.py
├── tasks/
│   ├── __init__.py
│   ├── task1_easy.py
│   ├── task2_medium.py
│   └── task3_hard.py
├── data/
│   ├── emails_easy.json
│   ├── emails_medium.json
│   └── emails_hard.json
└── graders/
    ├── __init__.py
    ├── grader_base.py
    ├── categorization_grader.py
    └── action_grader.py
```

## OpenEnv Spec Compliance

This environment implements the full OpenEnv specification with:
- ✅ Typed Pydantic models for observations, actions, and rewards
- ✅ step() and reset() methods
- ✅ state() method for environment introspection
- ✅ openenv.yaml with complete metadata
- ✅ Validated with `openenv validate`

## Validation

To validate the environment:

```bash
openenv validate openenv.yaml
```

## License

MIT