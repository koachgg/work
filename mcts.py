import random
import logging
from typing import List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

class ReasoningNode:
    def __init__(self, state: str, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        
class MonteCarloTreeSearch:
    def __init__(self, root, knowledge_base, evaluator):
        self.root = root
        self.kb = knowledge_base
        self.evaluator = evaluator
        
    def select(self, node: ReasoningNode) -> ReasoningNode:
        while node.children:
            if any(child.visits == 0 for child in node.children):
                return random.choice([child for child in node.children if child.visits == 0])
            
            ucb_values = [
                (child.value / child.visits) + 
                np.sqrt(2 * np.log(node.visits) / child.visits)
                for child in node.children
            ]
            return node.children[np.argmax(ucb_values)]
        return node
        
    def expand(self, node: ReasoningNode, actions: List[Callable]) -> None:
        try:
            context = self.kb.retrieve(node.state)
            
            for action in actions:
                try:
                    # All actions now accept both query and context
                    result = action(node.state, context)
                    if result:
                        child = ReasoningNode(result, parent=node)
                        node.children.append(child)
                except Exception as e:
                    logger.error(f"Error executing action: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in expand: {str(e)}")
            
    def simulate(self, node: ReasoningNode) -> float:
        return self.evaluator.evaluate(node.state)
        
    def backpropagate(self, node: ReasoningNode, reward: float) -> None:
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent