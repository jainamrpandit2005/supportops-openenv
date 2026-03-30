from abc import ABC, abstractmethod
from typing import Dict, Any


class GraderBase(ABC):
    """Base class for all task graders"""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def grade(self, state: Dict[str, Any]) -> float:
        """
        Grade the agent's performance on the task.
        
        Args:
            state: The final state of the environment after episode completion
        
        Returns:
            A float score between 0.0 and 1.0
        """
        pass