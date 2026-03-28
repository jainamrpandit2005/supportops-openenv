from __future__ import annotations
from copy import deepcopy
from typing import Optional, Dict
from app.models import Action, Reward, Observation, StepResult
from app.tasks import TASKS
from app.graders import grade_episode
from app.utils import build_observation

class SupportOpsEnv:
    def __init__(self, task_id: str = "easy_billing_refund"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.task_id = task_id
        self._state = None  # Internal state

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id:
            if task_id not in TASKS:
                raise ValueError(f"Unknown task_id: {task_id}")
            self.task_id = task_id

        task = deepcopy(TASKS[self.task_id])
        self._state = task  # Store task state
        self._state["current_step"] = 0
        self._state["done"] = False
        self._state["action_history"] = []
        self._state["reward_accumulator"] = 0.0
        self._state["visible_ticket_loaded"] = False
        return build_observation(self._state)

    def state(self) -> dict:
        if not self._state:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def _reward(self, delta: Dict[str, float], explanation: str) -> Reward:
        total = sum(delta.values())
        self._state["reward_accumulator"] += total
        return Reward(value=round(total, 4), components=delta, explanation=explanation)

    def _mark_done(self):
        self._state["done"] = True

    def step(self, action: Action) -> StepResult:
        if not self._state:
            raise RuntimeError("Call reset() first")
        if self._state["done"]:
            raise RuntimeError("Episode done. Call reset().")

        self._state["current_step"] += 1
        self._state["action_history"].append(action.dict())

        delta = {}
        explanation = "Action processed."
        delta["step_cost"] = -0.01  # small penalty

        # Example action handling
        if action.action_type == "read_ticket":
            if not self._state["visible_ticket_loaded"]:
                self._state["visible_ticket_loaded"] = True
                delta["read"] = 0.1
                explanation = "Ticket opened"
            else:
                delta["redundant_read"] = -0.02
                explanation = "Ticket already opened"

        elif action.action_type == "classify_ticket":
            if action.category is None:
                delta["invalid"] = -0.08
                explanation = "Missing category"
            else:
                self._state["predicted_category"] = action.category
                if action.category == self._state.get("gt_category"):
                    delta["classification"] = 0.2
                    explanation = "Correct classification"
                else:
                    delta["classification"] = -0.1
                    explanation = "Incorrect classification"

        elif action.action_type == "noop":
            delta["noop"] = -0.05
            explanation = "No-op discouraged"

        # Timeout check
        if self._state["current_step"] >= self._state["max_steps"]:
            self._mark_done()
            delta["timeout"] = -0.1
            explanation += " Max steps reached"

        reward = self._reward(delta, explanation)
        observation = build_observation(self._state)

        info = {}
        if self._state["done"]:
            info["grader"] = grade_episode(self._state)

        return StepResult(
            observation=observation,
            reward=reward,
            done=self._state["done"],
            info=info
        )