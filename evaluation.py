from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

class Evaluator:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
    def evaluate_reasoning_chain(self, chain: List[Dict]) -> float:
        if not chain:
            return 0.0
            
        scores = []
        for i in range(len(chain) - 1):
            coherence = self.compute_coherence(chain[i]['state'], chain[i+1]['state'])
            scores.append(coherence)
            
        return np.mean(scores) if scores else 0.0
        
    def compute_coherence(self, state1: str, state2: str) -> float:
        emb1 = self.model.encode([state1])[0]
        emb2 = self.model.encode([state2])[0]
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
    def self_consistency(self, answers: List[str], question: str) -> str:
        if not answers:
            return None
            
        answer_embeddings = self.model.encode(answers)
        question_embedding = self.model.encode([question])[0]
        
        answer_similarities = np.dot(answer_embeddings, answer_embeddings.T)
        question_relevance = np.dot(answer_embeddings, question_embedding)
        
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
        clusters = clustering.fit_predict(answer_embeddings)
        
        unique_clusters, cluster_sizes = np.unique(clusters, return_counts=True)
        largest_cluster = unique_clusters[np.argmax(cluster_sizes)]
        cluster_mask = clusters == largest_cluster
        
        cluster_answers_relevance = question_relevance[cluster_mask]
        best_in_cluster = np.argmax(cluster_answers_relevance)
        
        cluster_indices = np.where(cluster_mask)[0]
        best_answer_idx = cluster_indices[best_in_cluster]
        
        return answers[best_answer_idx]