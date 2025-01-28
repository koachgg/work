# import numpy as np

# class Node:
#     def __init__(self, state, parent=None):
#         self.state = state
#         self.parent = parent
#         self.children = []
#         self.visits = 0
#         self.q_value = 0

# class MonteCarloTreeSearch:
#     def __init__(self, root, max_depth=10):
#         self.root = root
#         self.max_depth = max_depth

#     def select(self, node):
#         if not node.children:
#             return node
#         return max(node.children, key=lambda x: x.q_value / (x.visits + 1e-8) + np.sqrt(2 * np.log(node.visits + 1) / (x.visits + 1e-8)))

#     def expand(self, node):
#         from actions import system_analysis, direct_answer, retrieval_answer, query_transformation, summary_answer
#         actions = [system_analysis, direct_answer, retrieval_answer, query_transformation, summary_answer]
#         for action in actions:
#             new_state = action(node.state)
#             node.children.append(Node(new_state, parent=node))


#     def simulate(self, node):
#         # Rollout to terminal state (max depth or answer found)
#         current_depth = 0
#         while current_depth < self.max_depth:
#             action = np.random.choice([system_analysis, direct_answer])
#             new_state = action(node.state)
#             current_depth += 1
#         return evaluate_answer(new_state)  # Placeholder for reward

#     def backpropagate(self, node, reward):
#         while node:
#             node.visits += 1
#             node.q_value += reward
#             node = node.parent

import numpy as np
from actions import system_analysis, direct_answer, retrieval_answer, query_transformation, summary_answer

class Node:
    def __init__(self, state, parent=None):
        self.state = state  # Current reasoning state (text)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.q_value = 0  # Reward value

class MonteCarloTreeSearch:
    def __init__(self, root, max_depth=10):
        self.root = root
        self.max_depth = max_depth

    def select(self, node):
        """
        Select the best child node using the UCT formula.
        """
        if not node.children:
            return node
        return max(node.children, key=lambda x: x.q_value / (x.visits + 1e-8) + np.sqrt(2 * np.log(node.visits + 1) / (x.visits + 1e-8)))

    def expand(self, node):
        """
        Expand the node by generating child nodes using reasoning actions.
        """
        actions = [system_analysis, direct_answer, retrieval_answer, query_transformation, summary_answer]
        for action in actions:
            new_state = action(node.state)
            node.children.append(Node(new_state, parent=node))

    def simulate(self, node):
        """
        Simulate a rollout from the current node to a terminal state.
        """
        current_depth = 0
        while current_depth < self.max_depth:
            # Randomly choose between system_analysis and direct_answer for simulation
            action = np.random.choice([system_analysis, direct_answer])
            new_state = action(node.state)
            current_depth += 1
        return self.evaluate_answer(new_state)  # Evaluate the final state

    def backpropagate(self, node, reward):
        """
        Backpropagate the reward from the terminal node to the root.
        """
        while node:
            node.visits += 1
            node.q_value += reward
            node = node.parent

    def evaluate_answer(self, answer):
        """
        Placeholder function to evaluate the quality of the answer.
        Replace this with a proper evaluation metric (e.g., similarity to ground truth).
        """
        # For now, return a random reward between 0 and 1
        return np.random.rand()