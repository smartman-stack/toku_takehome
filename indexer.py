"""
Legacy indexer module - maintained for backward compatibility.
For new implementations, use retriever.py with HybridRetriever.

This module provides basic FAISS indexing with sentence transformers.
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from data_loader import Document


class Indexer:
    """
    Basic vector indexer using sentence transformers and FAISS.
    
    Note: For production use, prefer HybridRetriever from retriever.py
    which provides:
    - Hybrid search (semantic + BM25)
    - Cross-encoder reranking
    - Query expansion
    - Caching
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize indexer with sentence transformer model.
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents: List[Document] = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
    
    def index_documents(self, documents: List[Document]):
        """
        Create embeddings and build FAISS index.
        
        Args:
            documents: List of Document objects to index
        """
        self.documents = documents
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index (Inner Product = Cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Indexed {len(documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if self.index is None:
            raise ValueError("Index not built. Call index_documents first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Build results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def get_document_count(self) -> int:
        """Get number of indexed documents."""
        return len(self.documents)
