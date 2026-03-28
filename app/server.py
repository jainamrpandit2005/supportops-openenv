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
def tasks():
    from app.tasks import TASKS
    return list(TASKS.keys())


@app.post("/reset")
def reset(task_id: str = "easy_billing_refund"):
    env = SupportOpsEnv(task_id=task_id)
    obs = env.reset()
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = env
    return env.reset(task_id)


@app.post("/step")
def step(action: Action):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    env = SESSIONS[session_id]
    result = env.step(action)
    return env.step(action)


@app.get("/state")
def state():
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    env = SESSIONS[session_id]
    return env.state()


@app.get("/grader")
def grader():
    if env._state is None:
        return {"error": "Env not initialized"}
    from app.graders import grade_episode
    return grade_episode(env._state)


@app.get("/baseline")
def baseline():
    return {"message": "Run your inference.py script for baseline evaluation."}