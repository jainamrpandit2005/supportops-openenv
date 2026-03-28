from fastapi import FastAPI, HTTPException
from typing import Optional
import uuid

from app.env import SupportOpsEnv
from app.models import Action
from app.tasks import TASKS
from app.graders import grade_episode
from app.baseline import run_baseline

app = FastAPI(title="SupportOps OpenEnv", version="1.0.0")

# Simple in-memory session store
SESSIONS = {}


@app.get("/")
def root():
    return {"status": "ok", "env": "SupportOps OpenEnv"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "difficulty": task["difficulty"],
                "max_steps": task["max_steps"],
                "action_schema": {
                    "action_type": [
                        "read_ticket",
                        "classify_ticket",
                        "set_priority",
                        "route_ticket",
                        "request_info",
                        "draft_reply",
                        "resolve_ticket",
                        "escalate_ticket",
                        "noop",
                    ],
                    "optional_fields": [
                        "category",
                        "priority",
                        "team",
                        "info_request",
                        "reply_text",
                        "notes",
                    ],
                },
            }
            for task in TASKS.values()
        ]
    }


@app.post("/reset")
def reset(task_id: Optional[str] = "easy_billing_refund"):
    env = SupportOpsEnv(task_id=task_id)
    obs = env.reset()
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = env
    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/step")
def step(session_id: str, action: Action):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    env = SESSIONS[session_id]
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    env = SESSIONS[session_id]
    return env.state().model_dump()


@app.get("/grader")
def grader(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    env = SESSIONS[session_id]
    st = env.state()
    if not st.done:
        return {"error": "Episode not finished yet."}
    return grade_episode(st)


@app.get("/baseline")
def baseline():
    return run_baseline()