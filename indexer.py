"""
Indexing module for creating embeddings and searchable index.
Uses sentence transformers for local embeddings (no API key required).
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
from data_loader import Document


class Indexer:
    """Creates and manages vector index for document retrieval."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize indexer with sentence transformer model.
        Using a lightweight model that works offline.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents: List[Document] = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
    
    def index_documents(self, documents: List[Document]):
        """Create embeddings and build FAISS index."""
        self.documents = documents
        
        # Generate embeddings for all documents
        texts = [doc.content for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype('float32'))
        
        print(f"Indexed {len(documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for similar documents.
        Returns list of (document, score) tuples.
        """
        if self.index is None:
            raise ValueError("Index not built. Call index_documents first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return documents with scores
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
