import os
import json
from typing import Dict, Any, List

from openai import OpenAI

from app.env import SupportOpsEnv
from app.tasks import TASKS
from app.models import Action


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


def choose_action_with_llm(client: OpenAI, observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses OpenAI-compatible client to choose the next action.
    Falls back to simple rule logic if response is invalid.
    """

    system_prompt = """
You are an AI customer support operations agent.
You must act inside a support ticket triage environment.

Your job is to:
- classify the ticket
- set priority
- route to the correct team
- request missing info if needed
- draft a helpful reply
- resolve or escalate appropriately

Return ONLY valid JSON in this format:
{
  "action_type": "read_ticket | classify_ticket | set_priority | route_ticket | request_info | draft_reply | resolve_ticket | escalate_ticket | noop",
  "category": "optional string",
  "priority": "optional string",
  "team": "optional string",
  "info_request": "optional string",
  "reply_text": "optional string",
  "notes": "optional string"
}
""".strip()

    user_prompt = f"""
Current observation:
{json.dumps(observation, indent=2)}

Choose the best next action.
Return only JSON.
""".strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )

        raw = completion.choices[0].message.content.strip()
        action = json.loads(raw)

        if "action_type" not in action:
            raise ValueError("Missing action_type")

        return action

    except Exception:
        # Safe fallback
        return {"action_type": "noop"}


def run_inference() -> Dict[str, Any]:
    """
    Runs all tasks using an OpenAI-compatible client and returns scores.
    """

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN if HF_TOKEN else "dummy-key"
    )

    env = SupportOpsEnv()

    results: List[Dict[str, Any]] = []

    for task_id, task in TASKS.items():
        obs = env.reset(task_id)
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < 12:
            action_dict = choose_action_with_llm(client, obs.model_dump())
            result = env.step(Action(**action_dict))
            obs = result.observation
            reward = result.reward.value if hasattr(result.reward, "value") else float(result.reward)
            done = result.done
            info = result.info

            total_reward += reward
            steps += 1

        results.append({
            "task_id": task_id,
            "task_name": getattr(task, "task_name", getattr(task, "name", task_id)),
            "reward_total": round(total_reward, 4),
            "steps": steps,
        })

    avg = sum(r["reward_total"] for r in results) / len(results) if results else 0.0

    return {
        "baseline_type": "openai_client_inference",
        "model_name": MODEL_NAME,
        "results": results,
        "average_score": avg,
    }


if __name__ == "__main__":
    output = run_inference()
    print(json.dumps(output, indent=2))