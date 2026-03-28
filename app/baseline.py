import json
from typing import Dict, Any

from app.env import SupportOpsEnv
from app.models import Action
from app.tasks import TASKS


def choose_actions_for_task(task_id: str):
    """
    Deterministic rule-based baseline actions for each task.
    This avoids needing OpenAI API and gives reproducible scores.
    """

    if task_id == "easy_billing_refund":
        return [
            Action(action_type="read_ticket"),
            Action(action_type="classify_ticket", category="billing"),
            Action(action_type="set_priority", priority="medium"),
            Action(action_type="route_ticket", team="billing_team"),
            Action(
                action_type="draft_reply",
                reply_text="We reviewed your duplicate charge and will help process a refund. Please confirm the billing issue."
            ),
            Action(action_type="resolve_ticket"),
        ]

    elif task_id == "medium_account_lockout":
        return [
            Action(action_type="read_ticket"),
            Action(action_type="classify_ticket", category="account"),
            Action(action_type="set_priority", priority="high"),
            Action(action_type="route_ticket", team="account_security"),
            Action(action_type="request_info", info_request="registered_email"),
            Action(action_type="request_info", info_request="last_successful_login_date"),
            Action(
                action_type="draft_reply",
                reply_text="We need to verify your identity using your email and last successful login date before we can securely restore access. Your case is being escalated to security."
            ),
            Action(action_type="escalate_ticket"),
        ]

    elif task_id == "hard_abuse_technical_combo":
        return [
            Action(action_type="read_ticket"),
            Action(action_type="classify_ticket", category="abuse"),
            Action(action_type="set_priority", priority="critical"),
            Action(action_type="route_ticket", team="trust_safety"),
            Action(action_type="request_info", info_request="workspace_id"),
            Action(action_type="request_info", info_request="affected_user_count"),
            Action(
                action_type="draft_reply",
                reply_text="This appears to be a serious security incident. We are initiating containment, reviewing suspicious API tokens, and escalating urgently to the incident response team."
            ),
            Action(action_type="escalate_ticket"),
        ]

    else:
        return [Action(action_type="noop")]


def run_single_task(task_id: str) -> Dict[str, Any]:
    env = SupportOpsEnv(task_id=task_id)
    env.reset()

    actions = choose_actions_for_task(task_id)
    final_result = None

    for action in actions:
        result = env.step(action)
        final_result = result
        if result.done:
            break

    return {
        "task_id": task_id,
        "task_name": TASKS[task_id]["task_name"],
        "reward_total": round(env.state().reward_accumulator, 4),
        "grader": final_result.info.get("grader", {}) if final_result else {},
        "steps": env.state().current_step,
    }


def run_baseline() -> Dict[str, Any]:
    results = []
    for task_id in TASKS.keys():
        results.append(run_single_task(task_id))

    valid_scores = [
        r["grader"]["score"] for r in results if r.get("grader") and "score" in r["grader"]
    ]
    avg_score = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0

    return {
        "baseline_type": "rule_based",
        "results": results,
        "average_score": avg_score,
    }


if __name__ == "__main__":
    print(json.dumps(run_baseline(), indent=2))