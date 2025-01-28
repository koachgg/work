from typing import List, Dict, Optional
import numpy as np

class ReasoningNode:
    def __init__(self, state: str, action_type: Optional[str] = None, parent=None):
        self.state = state
        self.action_type = action_type
        self.parent = parent
        self.children = []
        self.visits = 0
        self.q_value = 0
        self.reasoning_chain = []
        
    def add_to_chain(self, reasoning_step: Dict):
        self.reasoning_chain.append({
            'action': self.action_type,
            'state': self.state,
            'step': reasoning_step
        })
        
    def get_full_chain(self) -> List[Dict]:
        chain = []
        current = self
        while current:
            if current.action_type:
                chain.append({
                    'action': current.action_type,
                    'state': current.state
                })
            current = current.parent
        return list(reversed(chain))