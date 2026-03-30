import json
import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from enum import Enum


class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    is_spam: bool = False
    true_category: str = "personal"  # work, personal, spam, urgent


class EmailObservation(BaseModel):
    inbox: List[Dict[str, Any]]
    current_email_id: Optional[str] = None
    current_email_content: Optional[str] = None
    email_count: int = 0
    processed_count: int = 0
    inbox_status: str = "normal"


class EmailAction(BaseModel):
    action_type: str
    email_id: Optional[str] = None
    category: Optional[str] = None
    folder: Optional[str] = None


class EmailReward(BaseModel):
    value: float
    reason: str


class EmailTriageEnv:
    """Email Triage OpenEnv Environment"""
    
    def __init__(self, task_difficulty: str = "easy"):
        self.task_difficulty = task_difficulty
        self.emails: List[Email] = []
        self.processed_emails: Dict[str, str] = {}
        self.current_email_idx = 0
        self.step_count = 0
        self.max_steps = 200
        self.task_dataset = f"data/emails_{task_difficulty}.json"
        self._load_task_data()
    
    def _load_task_data(self):
        """Load task-specific email data"""
        if os.path.exists(self.task_dataset):
            with open(self.task_dataset, 'r') as f:
                data = json.load(f)
                self.emails = [Email(**email) for email in data]
        else:
            # Generate dummy emails if data file doesn't exist
            self.emails = self._generate_dummy_emails()
    
    def _generate_dummy_emails(self) -> List[Email]:
        """Generate dummy emails for testing"""
        emails = [
            Email(
                id="email_1",
                sender="boss@company.com",
                subject="Project deadline update",
                body="The project deadline has been moved to next Friday.",
                timestamp="2024-01-15T09:00:00",
                true_category="work"
            ),
            Email(
                id="email_2",
                sender="friend@gmail.com",
                subject="Let's grab coffee",
                body="Hey! Want to grab coffee this weekend?",
                timestamp="2024-01-15T10:30:00",
                true_category="personal"
            ),
            Email(
                id="email_3",
                sender="noreply@spam.com",
                subject="Click here to win prizes!",
                body="You have won 1 million dollars. Click here to claim.",
                timestamp="2024-01-15T11:00:00",
                is_spam=True,
                true_category="spam"
            ),
        ]
        return emails
    
    def reset(self) -> EmailObservation:
        """Reset the environment to initial state"""
        self.processed_emails = {}
        self.current_email_idx = 0
        self.step_count = 0
        return self._get_observation()
    
    def step(self, action: EmailAction) -> tuple[EmailObservation, float, bool, Dict]:
        """Execute one step in the environment"""
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}
        
        if action.action_type == "read":
            if action.email_id and action.email_id not in self.processed_emails:
                info["status"] = "email_read"
                reward = 0.1
        
        elif action.action_type == "categorize":
            if action.email_id and action.category:
                # Find the email and check if categorization is correct
                email = next((e for e in self.emails if e.id == action.email_id), None)
                if email:
                    is_correct = email.true_category == action.category
                    reward = 0.5 if is_correct else -0.2
                    self.processed_emails[action.email_id] = action.category
                    info["status"] = "email_categorized"
                    info["correct"] = is_correct
        
        elif action.action_type == "mark_spam":
            if action.email_id:
                email = next((e for e in self.emails if e.id == action.email_id), None)
                if email and email.is_spam:
                    reward = 0.7
                    self.processed_emails[action.email_id] = "spam"
                    info["status"] = "spam_marked"
                else:
                    reward = -0.3  # False positive
        
        elif action.action_type == "mark_urgent":
            if action.email_id:
                email = next((e for e in self.emails if e.id == action.email_id), None)
                if email and email.true_category == "work":
                    reward = 0.5
                    self.processed_emails[action.email_id] = "urgent"
        
        elif action.action_type == "noop":
            reward = 0.0
            info["status"] = "no_operation"
        
        # Check if we've processed all emails
        if len(self.processed_emails) >= len(self.emails) or self.step_count >= self.max_steps:
            done = True
            info["final_score"] = self._calculate_final_score()
        
        return self._get_observation(), reward, done, info
    
    def state(self) -> Dict[str, Any]:
        """Get current environment state"""
        return {
            "processed_emails": self.processed_emails,
            "total_emails": len(self.emails),
            "step_count": self.step_count,
            "task_difficulty": self.task_difficulty,
        }
    
    def _get_observation(self) -> EmailObservation:
        """Generate current observation"""
        current_email = None
        current_content = None
        
        if self.current_email_idx < len(self.emails):
            current_email = self.emails[self.current_email_idx]
            current_content = f"From: {current_email.sender}\nSubject: {current_email.subject}\n\n{current_email.body}"
            self.current_email_idx += 1
        
        inbox = [
            {
                "id": email.id,
                "sender": email.sender,
                "subject": email.subject,
                "processed": email.id in self.processed_emails
            }
            for email in self.emails
        ]
        
        return EmailObservation(
            inbox=inbox,
            current_email_id=current_email.id if current_email else None,
            current_email_content=current_content,
            email_count=len(self.emails),
            processed_count=len(self.processed_emails),
            inbox_status="normal"
        )
    
    def _calculate_final_score(self) -> float:
        """Calculate final score based on categorization accuracy"""
        if not self.emails:
            return 0.0
        
        correct = 0
        for email in self.emails:
            if email.id in self.processed_emails:
                if self.processed_emails[email.id] == email.true_category:
                    correct += 1
        
        return correct / len(self.emails)