from typing import Dict
from app.models import InternalState


def keyword_overlap_score(text: str, required_keywords: list[str]) -> float:
    if not text or not required_keywords:
        return 0.0
    text_l = text.lower()
    matched = sum(1 for kw in required_keywords if kw.lower() in text_l)
    return matched / len(required_keywords)


def grade_episode(state: InternalState) -> Dict:
    """
    Deterministic final grader score from 0.0 to 1.0
    """
    score = 0.0
    breakdown = {}

    category_score = 1.0 if state.predicted_category == state.gt_category else 0.0
    priority_score = 1.0 if state.predicted_priority == state.gt_priority else 0.0
    route_score = 1.0 if state.routed_team == state.gt_team else 0.0

    requested_info_set = set([x.strip().lower() for x in state.requested_info])
    required_info_set = set([x.strip().lower() for x in state.gt_required_info])

    if len(required_info_set) == 0:
        info_score = 1.0
    else:
        info_score = len(requested_info_set & required_info_set) / len(required_info_set)

    resolution_score = 0.0
    if state.gt_resolution == "resolve" and state.resolution_status == "resolved":
        resolution_score = 1.0
    elif state.gt_resolution == "escalate" and state.resolution_status == "escalated":
        resolution_score = 1.0

    reply_score = keyword_overlap_score(state.drafted_reply or "", state.gt_reply_keywords)

    breakdown["category"] = round(category_score, 3)
    breakdown["priority"] = round(priority_score, 3)
    breakdown["routing"] = round(route_score, 3)
    breakdown["info_gathering"] = round(info_score, 3)
    breakdown["resolution"] = round(resolution_score, 3)
    breakdown["reply_quality"] = round(reply_score, 3)

    # Weighted score
    score = (
        0.20 * category_score
        + 0.15 * priority_score
        + 0.15 * route_score
        + 0.15 * info_score
        + 0.20 * resolution_score
        + 0.15 * reply_score
    )

    # Penalty for wasting too many steps
    efficiency_penalty = max(0.0, (state.current_step - 6) * 0.01)
    score = max(0.0, min(1.0, score - efficiency_penalty))
    breakdown["efficiency_penalty"] = round(-efficiency_penalty, 3)

    return {
        "score": round(score, 4),
        "breakdown": breakdown,
    }