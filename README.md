---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - ai-agents
  - email-triage
  - reinforcement-learning
  - evaluation
---

# 📧 Email Triage OpenEnv

A **real-world OpenEnv environment** for training and evaluating AI agents on **email triage, categorization, prioritization, and inbox management**.

---

## 🚀 Overview

**Email Triage OpenEnv** simulates a realistic inbox management workflow where an AI agent must:

- 📩 Read incoming emails
- 🏷️ Categorize them correctly
- ⚠️ Detect spam
- 🔥 Mark urgent messages
- 🗂️ Archive or organize emails intelligently

This is a **real-world productivity and support workflow**, not a toy environment.

---

## 🎯 Why This Environment Matters

Email management is a common and high-value task performed by:

- 👨‍💼 Office professionals  
- 🧑‍💻 Customer support teams  
- 🏢 Businesses and operations teams  
- 🤖 AI assistants and productivity agents  

This environment helps evaluate whether an agent can:

- Understand **natural language**
- Make **context-aware decisions**
- Perform **task sequencing**
- Handle **prioritization under constraints**

---

# 🧠 OpenEnv Compliance

This project implements the required **OpenEnv interface**:

- ✅ `reset()`
- ✅ `step(action)`
- ✅ `state()`
- ✅ Typed **Pydantic models**
- ✅ `openenv.yaml`
- ✅ Deterministic tasks and graders
- ✅ Baseline inference script
- ✅ Hugging Face Docker deployment ready

---

# 🧩 Environment Design

## 👀 Observation Space

The agent receives inbox state and the currently focused email.

```python
class EmailObservation(BaseModel):
    inbox: List[Email]
    current_email_id: Optional[str]
    current_email_content: Optional[str]
    email_count: int
    processed_count: int
    inbox_status: str
