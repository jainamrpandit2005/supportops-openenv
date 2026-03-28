from app.models import Ticket


TASKS = {
    "easy_billing_refund": {
        "task_id": "easy_billing_refund",
        "task_name": "Refund Request Triage",
        "difficulty": "easy",
        "max_steps": 8,
        "ticket": Ticket(
            ticket_id="TKT-1001",
            customer_name="Aarav Patel",
            subject="Charged twice for my monthly plan",
            body=(
                "Hi team, I was charged twice this month for the Pro plan. "
                "I only have one subscription. Please help refund the duplicate charge."
            ),
            attachments=["billing_screenshot.png"],
            customer_tier="pro",
            account_age_days=240,
            previous_tickets=1,
        ),
        "gt_category": "billing",
        "gt_priority": "medium",
        "gt_team": "billing_team",
        "gt_required_info": [],
        "gt_resolution": "resolve",
        "gt_reply_keywords": ["refund", "duplicate charge", "billing", "confirm"],
    },

    "medium_account_lockout": {
        "task_id": "medium_account_lockout",
        "task_name": "Account Lockout with Missing Info",
        "difficulty": "medium",
        "max_steps": 10,
        "ticket": Ticket(
            ticket_id="TKT-2001",
            customer_name="Nisha Shah",
            subject="Can't log in after changing phone",
            body=(
                "Hello, I switched phones and now I can't receive my login verification code. "
                "I need access urgently because I have client files in the account."
            ),
            attachments=[],
            customer_tier="enterprise",
            account_age_days=800,
            previous_tickets=0,
        ),
        "gt_category": "account",
        "gt_priority": "high",
        "gt_team": "account_security",
        "gt_required_info": ["registered_email", "last_successful_login_date"],
        "gt_resolution": "escalate",
        "gt_reply_keywords": ["verify", "identity", "email", "security", "escalate"],
    },

    "hard_abuse_technical_combo": {
        "task_id": "hard_abuse_technical_combo",
        "task_name": "Abuse + Platform Incident Hybrid",
        "difficulty": "hard",
        "max_steps": 12,
        "ticket": Ticket(
            ticket_id="TKT-3001",
            customer_name="Rohan Mehta",
            subject="Suspicious account activity and mass email sending",
            body=(
                "My team noticed thousands of emails were sent from our workspace overnight, "
                "and some users are now locked out. We also see unknown API tokens created. "
                "Please stop this immediately. This may be a security breach."
            ),
            attachments=["audit_log.csv", "screenshot_tokens.png"],
            customer_tier="enterprise",
            account_age_days=1200,
            previous_tickets=3,
        ),
        "gt_category": "abuse",
        "gt_priority": "critical",
        "gt_team": "trust_safety",
        "gt_required_info": ["workspace_id", "affected_user_count"],
        "gt_resolution": "escalate",
        "gt_reply_keywords": [
            "security", "incident", "containment", "tokens", "escalate", "urgent"
        ],
    },
}