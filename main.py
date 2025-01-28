import logging
import time
from knowledge_base import KnowledgeBase
from action_manager import ActionManager
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_airrag(question: str):
    try:
        logger.info("Initializing components...")
        kb = KnowledgeBase()
        action_manager = ActionManager()
        
        logger.info("Fetching knowledge...")
        # Get knowledge
        terms = kb.extract_search_terms(question)
        for term in terms:
            kb.add_knowledge(term)
        
        # Get relevant context
        context = kb.retrieve(question, k=3)
        if not context:
            return "Unable to find relevant information to answer the question."

        # Generate answer using retrieval-based approach
        logger.info("Generating answer...")
        answer = action_manager.retrieval_answer(question, context)
        
        return answer if answer else "Unable to generate a response."

    except Exception as e:
        logger.error(f"Error in run_airrag: {str(e)}", exc_info=True)
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    questions = [
        "What is the capital of France and what is its population?",
        "Who invented the telephone and in which year?",
        "What is the height of Mount Everest?"
    ]
    
    for question in questions:
        print(f"\nProcessing question: {question}")
        start_time = time.time()
        answer = run_airrag(question)
        end_time = time.time()
        print(f"Answer: {answer}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")