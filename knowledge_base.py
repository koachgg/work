import wikipediaapi
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List
import os
import json
import re

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self):
        # Initialize the sentence transformer
        self.retriever = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.embedding_dim = self.retriever.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Initialize Wikipedia API
        user_agent = "AirRAG-Research-Bot/1.0"
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        
        self.texts = []
        self.cache_dir = "data/knowledge_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def clean_query(self, query: str) -> str:
        return re.sub(r'[?!.,]', '', query).strip()

    def extract_search_terms(self, query: str) -> List[str]:
        stop_words = {'what', 'is', 'the', 'where', 'when', 'who', 'how', 'why', 
                     'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        terms = self.clean_query(query.lower()).split()
        meaningful_terms = []
        
        i = 0
        while i < len(terms):
            if i + 1 < len(terms):
                combined = f"{terms[i]} {terms[i+1]}"
                if not any(word in stop_words for word in combined.split()):
                    meaningful_terms.append(combined)
                    i += 2
                    continue
            
            if terms[i] not in stop_words:
                meaningful_terms.append(terms[i])
            i += 1
            
        return meaningful_terms if meaningful_terms else [query]

    def get_wiki_content(self, query: str) -> List[str]:
        """Get content from Wikipedia with better handling."""
        try:
            # Clean the query and try to get the page
            clean_query = self.clean_query(query.lower())
            page = self.wiki.page(clean_query)
            
            if page.exists():
                # Split into paragraphs and filter
                paragraphs = [p.strip() for p in page.text.split('\n\n') 
                            if len(p.strip()) > 50][:3]
                return [p[:300] for p in paragraphs]  # Limit length
            
            # If page doesn't exist, try individual terms
            terms = self.extract_search_terms(query)
            for term in terms:
                page = self.wiki.page(term)
                if page.exists():
                    paragraphs = [p.strip() for p in page.text.split('\n\n') 
                                if len(p.strip()) > 50][:2]
                    return [p[:300] for p in paragraphs]
            
            return []

        except Exception as e:
            logger.error(f"Error in get_wiki_content: {str(e)}")
            return []
    

    def add_knowledge(self, query: str):
        try:
            cache_file = os.path.join(self.cache_dir, f"{hash(query)}.json")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    self.texts.extend(cached_data['texts'])
                    embeddings = np.array(cached_data['embeddings'], dtype='float32')
                    self.index.add(embeddings)
                    return
            
            paragraphs = self.get_wiki_content(query)
            
            if paragraphs:
                embeddings = self.retriever.encode(paragraphs, convert_to_numpy=True)
                self.index.add(embeddings.astype('float32'))
                self.texts.extend(paragraphs)
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'texts': paragraphs,
                        'embeddings': embeddings.tolist()
                    }, f, ensure_ascii=False)
                    
        except Exception as e:
            logger.error(f"Error in add_knowledge: {str(e)}")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        try:
            if not self.texts:
                return []
                
            query_embedding = self.retriever.encode([query], convert_to_numpy=True)
            D, I = self.index.search(query_embedding.astype('float32'), min(k, len(self.texts)))
            
            return [self.texts[i] for i in I[0] if i < len(self.texts)]
            
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            return []