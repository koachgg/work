
import logging
import json
from knowledge_base import KnowledgeBase
from action_manager import ActionManager
from mcts import MonteCarloTreeSearch, ReasoningNode
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str):
    """
    Load QA dataset and preprocess it.
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return [(item["question"], item["answer"]) for item in data]


def evaluate(queries: List[str], answers: List[str], kb, am):
    """
    Evaluate the system on a QA dataset.
    """
    correct = 0
    total = len(queries)

    for query, answer in zip(queries, answers):
        kb.add_knowledge(query)
        context = kb.retrieve(query, k=3)

        # Initialize MCTS
        root_node = ReasoningNode(query)
        mcts = MonteCarloTreeSearch(root_node, kb, am)

        # Define actions (from action_manager)
        actions = [
            am.system_analysis,
            am.query_transformation,
            am.retrieval_answer,
            am.summary_answer
        ]

        # Run MCTS
        best_path = mcts.run(max_iterations=20, actions=actions)

        # Check final answer
        prediction = best_path.split(" -> ")[-1].strip()
        if prediction.lower() == answer.lower():
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    # Load dataset (e.g., HotpotQA)
    dataset_path = "hotpotqa.json"
    dataset = load_dataset(dataset_path)
    queries, answers = zip(*dataset)

    # Initialize components
    kb = KnowledgeBase()
    am = ActionManager()

    # Evaluate on the dataset
    evaluate(queries, answers, kb, am)
