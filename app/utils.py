from app.models import InternalState, Observation


def summarize_history(action_history: list[dict]) -> list[str]:
    summaries = []
    for i, item in enumerate(action_history[-5:], start=1):
        a = item.get("action_type", "unknown")
        summaries.append(f"{i}. {a}")
    return summaries


def build_observation(state: InternalState) -> Observation:
    visible_ticket = state.ticket if state.visible_ticket_loaded else None

    required_missing = []
    for req in state.gt_required_info:
        if req not in state.requested_info:
            required_missing.append(req)

    return Observation(
        task_id=state.task_id,
        task_name=state.task_name,
        difficulty=state.difficulty,
        step=state.current_step,
        max_steps=state.max_steps,
        visible_ticket=visible_ticket,
        current_labels={
            "predicted_category": state.predicted_category,
            "predicted_priority": state.predicted_priority,
            "routed_team": state.routed_team,
            "resolution_status": state.resolution_status,
        },
        outstanding_missing_info=required_missing,
        action_history_summary=summarize_history(state.action_history),
        done=state.done,
    )