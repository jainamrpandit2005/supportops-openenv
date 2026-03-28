from __future__ import annotations
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field


TicketCategory = Literal[
    "billing", "technical", "account", "shipping", "abuse", "unknown"
]

PriorityLevel = Literal["low", "medium", "high", "critical"]

TeamName = Literal[
    "billing_team",
    "tech_support",
    "account_security",
    "logistics",
    "trust_safety",
    "general_queue"
]

ActionType = Literal[
    "read_ticket",
    "classify_ticket",
    "set_priority",
    "route_ticket",
    "request_info",
    "draft_reply",
    "resolve_ticket",
    "escalate_ticket",
    "noop"
]


class Ticket(BaseModel):
    ticket_id: str
    customer_name: str
    subject: str
    body: str
    attachments: List[str] = Field(default_factory=list)
    customer_tier: Literal["free", "pro", "enterprise"] = "free"
    account_age_days: int = 30
    previous_tickets: int = 0
    language: str = "en"


class InternalState(BaseModel):
    task_id: str
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int
    current_step: int = 0
    done: bool = False

    ticket: Ticket

    visible_ticket_loaded: bool = False
    predicted_category: Optional[TicketCategory] = None
    predicted_priority: Optional[PriorityLevel] = None
    routed_team: Optional[TeamName] = None
    requested_info: List[str] = Field(default_factory=list)
    drafted_reply: Optional[str] = None
    resolution_status: Optional[Literal["resolved", "escalated"]] = None

    action_history: List[Dict] = Field(default_factory=list)
    reward_accumulator: float = 0.0

    # Ground truth (hidden from agent)
    gt_category: TicketCategory
    gt_priority: PriorityLevel
    gt_team: TeamName
    gt_required_info: List[str] = Field(default_factory=list)
    gt_resolution: Literal["resolve", "escalate"] = "resolve"
    gt_reply_keywords: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    step: int
    max_steps: int
    visible_ticket: Optional[Ticket] = None

    current_labels: Dict[str, Optional[str]]
    outstanding_missing_info: List[str]
    action_history_summary: List[str]
    done: bool


class Action(BaseModel):
    action_type: ActionType
    category: Optional[TicketCategory] = None
    priority: Optional[PriorityLevel] = None
    team: Optional[TeamName] = None
    info_request: Optional[str] = None
    reply_text: Optional[str] = None
    notes: Optional[str] = None


class Reward(BaseModel):
    value: float
    components: Dict[str, float]
    explanation: str


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict