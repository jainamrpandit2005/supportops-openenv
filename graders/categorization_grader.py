from typing import Dict, Any
from graders.grader_base import GraderBase


class CategorizationGrader(GraderBase):
    """Grader for basic email categorization task (Easy)"""
    
    def grade(self, state: Dict[str, Any]) -> float:
        """
        Grade based on categorization accuracy.
        
        Args:
            state: Environment state containing processed emails and accuracy info
        
        Returns:
            Score between 0.0 and 1.0 representing categorization accuracy
        """
        if "final_score" not in state:
            return 0.0
        
        # Scale the final score with some tolerance
        score = state["final_score"]
        
        # Apply some adjustments
        # Minimum 50% to get any credit, max 1.0 for perfect
        if score < 0.5:
            return 0.0
        
        # Linear scale from 0.5 (returns 0) to 1.0 (returns 1.0)
        return (score - 0.5) * 2.0