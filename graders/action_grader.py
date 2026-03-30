from typing import Dict, Any
from graders.grader_base import GraderBase


class ActionGrader(GraderBase):
    """Grader for email prioritization task (Medium)"""
    
    def grade(self, state: Dict[str, Any]) -> float:
        """
        Grade based on categorization accuracy and appropriate action selection.
        
        Args:
            state: Environment state with processed emails and actions taken
        
        Returns:
            Score between 0.0 and 1.0 based on accuracy and action quality
        """
        if "final_score" not in state:
            return 0.0
        
        base_score = state.get("final_score", 0.0)
        action_quality = state.get("action_quality", 0.5)
        
        # Combined score: 70% categorization accuracy, 30% action selection
        combined = (base_score * 0.7) + (action_quality * 0.3)
        
        return min(combined, 1.0)


class ComplexGrader(GraderBase):
    """Grader for complex inbox management task (Hard)"""
    
    def grade(self, state: Dict[str, Any]) -> float:
        """
        Grade based on comprehensive inbox management including pattern recognition.
        
        Args:
            state: Environment state with comprehensive metrics
        
        Returns:
            Score between 0.0 and 1.0
        """
        if "final_score" not in state:
            return 0.0
        
        base_score = state.get("final_score", 0.0)
        spam_accuracy = state.get("spam_detection_accuracy", 0.5)
        pattern_recognition = state.get("pattern_recognition_score", 0.5)
        
        # Complex task requires:
        # 50% base categorization accuracy
        # 30% spam detection accuracy
        # 20% pattern recognition
        
        combined = (base_score * 0.5) + (spam_accuracy * 0.3) + (pattern_recognition * 0.2)
        
        return min(combined, 1.0)