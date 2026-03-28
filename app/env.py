from __future__ import annotations
from typing import Dict, Optional
from copy import deepcopy

from app.models import (
    Action,
    Reward,
    StepResult,
    InternalState,
)
from app.tasks import TASKS
from app.utils import build_observation
from app.graders import grade_episode


class SupportOpsEnv:
    def __init__(self, task_id: str = "easy_billing_refund"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.task_id = task_id
        self._state: Optional[InternalState] = None

    def reset(self, task_id: Optional[str] = None):
        if task_id is not None:
            if task_id not in TASKS:
                raise ValueError(f"Unknown task_id: {task_id}")
            self.task_id = task_id

        task = deepcopy(TASKS[self.task_id])
        self._state = InternalState(**task)
        return build_observation(self._state)

    def state(self) -> InternalState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def _reward(self, delta: Dict[str, float], explanation: str) -> Reward:
        total = sum(delta.values())
        self._state.reward_accumulator += total
        return Reward(value=round(total, 4), components=delta, explanation=explanation)

    def _mark_done(self):
        self._state.done = True

    def step(self, action: Action) -> StepResult:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode already done. Call reset().")

        self._state.current_step += 1
        self._state.action_history.append(action.model_dump())

        delta = {}
        explanation = "Action processed."

        # Small step penalty to encourage efficiency
        delta["step_cost"] = -0.01

        if action.action_type == "read_ticket":
            if not self._state.visible_ticket_loaded:
                self._state.visible_ticket_loaded = True
                delta["read"] = 0.10
                explanation = "Ticket opened successfully."
            else:
                delta["redundant_read"] = -0.02
                explanation = "Ticket already opened."

        elif action.action_type == "classify_ticket":
            if action.category is None:
                delta["invalid"] = -0.08
                explanation = "Missing category."
            else:
                self._state.predicted_category = action.category
                if action.category == self._state.gt_category:
                    delta["classification"] = 0.20
                    explanation = "Correct classification."
                elif action.category == "unknown":
                    delta["classification"] = 0.02
                    explanation = "Fallback classification used."
                else:
                    delta["classification"] = -0.10
                    explanation = "Incorrect classification."

        elif action.action_type == "set_priority":
            if action.priority is None:
                delta["invalid"] = -0.08
                explanation = "Missing priority."
            else:
                self._state.predicted_priority = action.priority
                if action.priority == self._state.gt_priority:
                    delta["priority"] = 0.18
                    explanation = "Correct priority set."
                else:
                    delta["priority"] = -0.10
                    explanation = "Incorrect priority."

        elif action.action_type == "route_ticket":
            if action.team is None:
                delta["invalid"] = -0.08
                explanation = "Missing team."
            else:
                self._state.routed_team = action.team
                if action.team == self._state.gt_team:
                    delta["routing"] = 0.18
                    explanation = "Correct routing."
                else:
                    delta["routing"] = -0.12
                    explanation = "Incorrect routing."

        elif action.action_type == "request_info":
            if not action.info_request:
                delta["invalid"] = -0.08
                explanation = "Missing requested info field."
            else:
                info_key = action.info_request.strip().lower()
                if info_key not in self._state.requested_info:
                    self._state.requested_info.append(info_key)
                    if info_key in [x.lower() for x in self._state.gt_required_info]:
                        delta["info_request"] = 0.12
                        explanation = f"Useful information requested: {info_key}"
                    else:
                        delta["info_request"] = -0.03
                        explanation = f"Non-essential information requested: {info_key}"
                else:
                    delta["duplicate_info_request"] = -0.04
                    explanation = "Duplicate information request."

        elif action.action_type == "draft_reply":
            if not action.reply_text:
                delta["invalid"] = -0.08
                explanation = "Missing reply text."
            else:
                self._state.drafted_reply = action.reply_text
                reply_text_l = action.reply_text.lower()
                hits = sum(
                    1 for kw in self._state.gt_reply_keywords if kw.lower() in reply_text_l
                )
                delta["reply"] = min(0.20, hits * 0.04)
                explanation = f"Draft reply updated with {hits} useful support cues."

        elif action.action_type == "resolve_ticket":
            self._state.resolution_status = "resolved"
            if self._state.gt_resolution == "resolve":
                delta["resolve"] = 0.25
                explanation = "Correctly resolved the ticket."
                self._mark_done()
            else:
                delta["resolve"] = -0.20
                explanation = "Incorrectly resolved a ticket that required escalation."
                self._mark_done()

        elif action.action_type == "escalate_ticket":
            self._state.resolution_status = "escalated"
            if self._state.gt_resolution == "escalate":
                delta["escalate"] = 0.25
                explanation = "Correctly escalated the ticket."
                self._mark_done()
            else:
                delta["escalate"] = -0.12
                explanation = "Escalation was unnecessary."
                self._mark_done()

        elif action.action_type == "noop":
            delta["noop"] = -0.05
            explanation = "No-op is discouraged."

        else:
            delta["unknown_action"] = -0.10
            explanation = "Unknown action."

        # Episode boundary
        if self._state.current_step >= self._state.max_steps and not self._state.done:
            self._mark_done()
            delta["timeout"] = -0.10
            explanation += " Max steps reached."

        reward = self._reward(delta, explanation)
        observation = build_observation(self._state)

        info = {}
        if self._state.done:
            info["grader"] = grade_episode(self._state)

        return StepResult(
            observation=observation,
            reward=reward,
            done=self._state.done,
            info=info,
        )