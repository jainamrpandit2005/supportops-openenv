from app.env import SupportOpsEnv
from app.models import Action


def test_reset_and_read():
    env = SupportOpsEnv("easy_billing_refund")
    obs = env.reset()
    assert obs.visible_ticket is None

    result = env.step(Action(action_type="read_ticket"))
    assert result.observation.visible_ticket is not None
    assert result.reward.value > 0


def test_easy_path():
    env = SupportOpsEnv("easy_billing_refund")
    env.reset()

    env.step(Action(action_type="read_ticket"))
    env.step(Action(action_type="classify_ticket", category="billing"))
    env.step(Action(action_type="set_priority", priority="medium"))
    env.step(Action(action_type="route_ticket", team="billing_team"))
    env.step(Action(action_type="draft_reply", reply_text="We will review your duplicate charge and process a refund. Please confirm."))
    result = env.step(Action(action_type="resolve_ticket"))

    assert result.done is True
    assert "grader" in result.info
    assert 0.0 <= result.info["grader"]["score"] <= 1.0