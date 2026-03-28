# SupportOps OpenEnv

A real-world OpenEnv environment for training and evaluating AI agents on customer support operations.

## Why this environment?

Most agent benchmarks are too synthetic:
- toy web tasks,
- game-like state transitions,
- trivial classification loops.

SupportOps models a genuine workflow used in real businesses:

- read support tickets,
- classify issue type,
- set urgency,
- route to the right team,
- request missing information,
- draft customer replies,
- decide whether to resolve or escalate.

This makes it useful for:
- agent evaluation,
- RLHF / RLAIF experimentation,
- workflow automation benchmarking,
- support/copilot research.

---

## Environment API

### `reset(task_id)`
Starts a new episode.

### `step(action)`
Takes one action and returns:
- observation
- reward
- done
- info

### `state()`
Returns the full internal environment state.

---

## Observation Space

Typed Pydantic model: `Observation`

Fields include:
- `task_id`
- `task_name`
- `difficulty`
- `step`
- `max_steps`
- `visible_ticket`
- `current_labels`
- `outstanding_missing_info`
- `action_history_summary`
- `done`

---

## Action Space

Typed Pydantic model: `Action`

### Supported `action_type`s
- `read_ticket`
- `classify_ticket`
- `set_priority`
- `route_ticket`
- `request_info`
- `draft_reply`
- `resolve_ticket`
- `escalate_ticket`
- `noop`

Optional fields:
- `category`
- `priority`
- `team`
- `info_request`
- `reply_text`
- `notes`

---

## Reward Design

The reward is shaped over the full trajectory:

### Positive signals
- Opening the ticket
- Correct classification
- Correct priority
- Correct routing
- Requesting genuinely missing information
- Drafting a useful customer reply
- Correct final resolution/escalation

### Negative signals
- Wasted steps
- Repeated reads
- Duplicate info requests
- Wrong classification/routing/priority
- Incorrect resolution decision
- No-op behavior
- Episode timeout

This provides dense learning signal instead of only sparse terminal reward.

---

## Tasks

### 1) Easy — `easy_billing_refund`
A duplicate billing charge refund request.

**Expected difficulty:** Easy  
**Goal:** Correctly classify, route, draft a billing reply, and resolve.

---

### 2) Medium — `medium_account_lockout`
A high-value account access issue with missing verification info.

**Expected difficulty:** Medium  
**Goal:** Recognize account security issue, request missing verification fields, route appropriately, and escalate.

---

### 3) Hard — `hard_abuse_technical_combo`
A potential enterprise security breach involving abuse and platform compromise.

**Expected difficulty:** Hard  
**Goal:** Detect incident severity, gather key missing fields, route to trust & safety, draft incident-aware reply, and escalate.

---

## Grading

Each episode is scored deterministically from **0.0 to 1.0**.

Grader dimensions:
- category correctness
- priority correctness
- routing correctness
- information gathering completeness
- resolution correctness
- reply quality keyword coverage
- efficiency penalty

---

## Setup

### Local install

```bash
git clone <your-repo-url>
cd supportops-openenv
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

## Baseline Inference

Runs a deterministic rule-based baseline agent (no external API required).

Run:

```bash
python -m app.baseline