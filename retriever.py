"""
Advanced retriever module with hybrid search (BM25 + semantic) and reranking.
Provides high-quality document retrieval for the RAG pipeline.
"""
import numpy as np
import faiss
import hashlib
import pickle
import os
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
from data_loader import Document
from config import get_config, Config
import math
import re


class BM25:
    """BM25 implementation for keyword-based retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.n_docs: int = 0
        self.documents: List[Document] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization with lowercasing and punctuation removal."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens and stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                     'between', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
                     'on', 'off', 'over', 'under', 'again', 'further', 'then',
                     'once', 'here', 'there', 'when', 'where', 'why', 'how',
                     'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                     'because', 'as', 'until', 'while', 'this', 'that', 'these',
                     'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        return [t for t in tokens if len(t) > 1 and t not in stopwords]
    
    def fit(self, documents: List[Document]):
        """Build BM25 index from documents."""
        self.documents = documents
        self.n_docs = len(documents)
        self.doc_term_freqs = []
        self.doc_lens = []
        self.doc_freqs = defaultdict(int)
        
        for doc in documents:
            # Combine content and keywords for indexing
            text = doc.content + " " + " ".join(doc.keywords)
            tokens = self._tokenize(text)
            self.doc_lens.append(len(tokens))
            
            # Term frequencies for this document
            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            self.doc_term_freqs.append(dict(term_freq))
            
            # Document frequencies
            for token in set(tokens):
                self.doc_freqs[token] += 1
        
        self.avg_doc_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        
        # Calculate IDF
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for documents matching query."""
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx, doc in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_lens[idx]
            term_freqs = self.doc_term_freqs[idx]
            
            for token in query_tokens:
                if token not in term_freqs:
                    continue
                
                tf = term_freqs[token]
                idf = self.idf.get(token, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * (numerator / denominator)
            
            scores.append((doc, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class HybridRetriever:
    """
    Hybrid retriever combining semantic search with BM25.
    Includes optional cross-encoder reranking for improved precision.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.config.model.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.model.embedding_model)
        
        # Initialize reranker if enabled
        self.reranker: Optional[CrossEncoder] = None
        if self.config.model.use_reranker:
            print(f"Loading reranker model: {self.config.model.reranker_model}")
            try:
                self.reranker = CrossEncoder(self.config.model.reranker_model)
            except Exception as e:
                print(f"Warning: Could not load reranker: {e}. Continuing without reranking.")
                self.reranker = None
        
        # Initialize indices
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25 = BM25()
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Cache
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._query_cache: Dict[str, List[Tuple[Document, float]]] = {}
    
    def _get_cache_path(self) -> str:
        """Get path for embedding cache."""
        cache_dir = self.config.cache.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "embeddings_cache.pkl")
    
    def _load_cache(self) -> bool:
        """Load embeddings from cache if available."""
        if not self.config.cache.enable_embedding_cache:
            return False
        
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    # Verify cache is valid for current documents
                    if cached.get('doc_hash') == self._get_docs_hash():
                        self.embeddings = cached['embeddings']
                        print("Loaded embeddings from cache")
                        return True
            except Exception as e:
                print(f"Cache load failed: {e}")
        return False
    
    def _save_cache(self):
        """Save embeddings to cache."""
        if not self.config.cache.enable_embedding_cache:
            return
        
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'doc_hash': self._get_docs_hash(),
                    'embeddings': self.embeddings
                }, f)
            print("Saved embeddings to cache")
        except Exception as e:
            print(f"Cache save failed: {e}")
    
    def _get_docs_hash(self) -> str:
        """Generate hash of document contents for cache validation."""
        content = "".join([d.content for d in self.documents])
        return hashlib.md5(content.encode()).hexdigest()
    
    def index_documents(self, documents: List[Document]):
        """Build both semantic and BM25 indices."""
        self.documents = documents
        
        # Build BM25 index
        print("Building BM25 index...")
        self.bm25.fit(documents)
        
        # Check cache for embeddings
        if not self._load_cache():
            # Generate embeddings
            print("Generating embeddings...")
            texts = [doc.content for doc in documents]
            self.embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            self._save_cache()
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print(f"Indexed {len(documents)} documents")
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform semantic search using FAISS."""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents) and idx >= 0:
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        if not self.config.retrieval.use_query_expansion:
            return query
        
        query_lower = query.lower()
        expansions = []
        
        # Domain-specific expansions
        expansion_map = {
            'recording': ['call recording', 'voice recording', 'record calls', 'meranti', 'pro'],
            'sso': ['single sign-on', 'SSO', 'authentication', 'raintree', 'plan'],
            'quota': ['limit', 'exceeded', 'usage', 'tier', 'whatsapp', 'conversations', 't1', 't2', 't3'],
            'whatsapp': ['WA', 'WhatsApp API', 'messaging', 'quota', 'tier', 'kapok'],
            'exceeded': ['quota', 'limit', 'tier', 'upgrade', 'whatsapp'],
            'discount': ['savings', 'reduction', 'offer', 'pricing', 'annual', 'nonprofit'],
            'plan': ['package', 'subscription', 'tier', 'pricing'],
            'upgrade': ['change plan', 'higher tier', 'better plan', 'pro'],
            'prepay': ['annual', 'yearly', 'advance payment', 'discount'],
            'nonprofit': ['non-profit', 'charity', 'NGO', 'discount'],
            'advise': ['recommend', 'suggest', 'plan', 'choice'],
            'choice': ['plan', 'option', 'recommend'],
        }
        
        for term, synonyms in expansion_map.items():
            if term in query_lower:
                expansions.extend(synonyms[:self.config.retrieval.max_expansion_terms])
        
        if expansions:
            return query + " " + " ".join(expansions)
        return query
    
    def _hybrid_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Combine semantic and BM25 search results."""
        expanded_query = self._expand_query(query)
        
        # Get results from both methods
        semantic_results = self._semantic_search(expanded_query, k * 2)
        bm25_results = self.bm25.search(expanded_query, k * 2)
        
        # Normalize scores
        semantic_scores = {doc.citation: score for doc, score in semantic_results}
        bm25_scores = {doc.citation: score for doc, score in bm25_results}
        
        # Normalize BM25 scores to 0-1 range
        if bm25_scores:
            max_bm25 = max(bm25_scores.values()) or 1
            bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}
        
        # Combine scores with weights
        combined: Dict[str, Tuple[Document, float]] = {}
        
        for doc, score in semantic_results:
            citation = doc.citation
            semantic_score = score * self.config.retrieval.semantic_weight
            bm25_score = bm25_scores.get(citation, 0) * self.config.retrieval.bm25_weight
            combined[citation] = (doc, semantic_score + bm25_score)
        
        for doc, score in bm25_results:
            citation = doc.citation
            if citation not in combined:
                semantic_score = semantic_scores.get(citation, 0) * self.config.retrieval.semantic_weight
                bm25_score = (score / (max(bm25_scores.values()) or 1)) * self.config.retrieval.bm25_weight
                combined[citation] = (doc, semantic_score + bm25_score)
        
        # Sort by combined score
        results = list(combined.values())
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def _rerank(self, query: str, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Rerank results using cross-encoder."""
        if not self.reranker or not results:
            return results
        
        # Prepare pairs for reranking
        pairs = [(query, doc.content) for doc, _ in results]
        
        # Get reranking scores
        try:
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine with original scores (weighted)
            reranked = []
            for (doc, orig_score), rerank_score in zip(results, rerank_scores):
                # Give more weight to reranker
                combined_score = 0.3 * orig_score + 0.7 * float(rerank_score)
                reranked.append((doc, combined_score))
            
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked
        except Exception as e:
            print(f"Reranking failed: {e}. Using original ranking.")
            return results
    
    def _boost_critical_documents(self, query: str, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Boost documents that are critical for certain query types."""
        query_lower = query.lower()
        boosted = list(results)
        citations_seen = {doc.citation for doc, _ in boosted}
        
        # Define critical document patterns for query types
        critical_docs = {
            'quota': ['wa_api_quota_tiers', 'quota'],
            'whatsapp': ['wa_api_quota_tiers', 'quota', 'kapok'],
            'exceeded': ['wa_api_quota_tiers', 'quota', 'tier'],
            'recording': ['call_recording', 'pro', 'features_matrix'],
            'sso': ['sso', 'raintree', 'features_matrix'],
        }
        
        # Find which critical patterns apply
        patterns_needed = set()
        for term, patterns in critical_docs.items():
            if term in query_lower:
                patterns_needed.update(patterns)
        
        if not patterns_needed:
            return results
        
        # Search all documents for critical ones not yet in results
        for doc in self.documents:
            if doc.citation in citations_seen:
                continue
            
            # Check if this document matches any needed pattern
            citation_lower = doc.citation.lower()
            section = doc.metadata.get('section', '').lower()
            
            for pattern in patterns_needed:
                if pattern in citation_lower or pattern in section:
                    # Add this critical document with a minimum score
                    boosted.append((doc, 0.1))  # Minimum score to ensure inclusion
                    citations_seen.add(doc.citation)
                    break
        
        return boosted
    
    def search(self, query: str, k: Optional[int] = None, 
               use_cache: bool = True) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return (default from config)
            use_cache: Whether to use query cache
        
        Returns:
            List of (document, score) tuples
        """
        if self.faiss_index is None:
            raise ValueError("Index not built. Call index_documents first.")
        
        k = k or self.config.retrieval.final_k
        
        # Check cache
        cache_key = f"{query}_{k}"
        if use_cache and self.config.cache.enable_query_cache:
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]
        
        # Get initial results with hybrid search
        initial_k = self.config.retrieval.initial_k
        results = self._hybrid_search(query, initial_k)
        
        # Rerank if enabled
        if self.reranker:
            results = self._rerank(query, results)
        
        # Boost critical documents for specific query types
        results = self._boost_critical_documents(query, results)
        
        # Filter by relevance threshold
        threshold = self.config.retrieval.relevance_threshold
        results = [(doc, score) for doc, score in results if score >= threshold]
        
        # Sort by score again after boosting
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k
        results = results[:k]
        
        # Cache results
        if use_cache and self.config.cache.enable_query_cache:
            self._query_cache[cache_key] = results
        
        return results
    
    def search_by_type(self, query: str, doc_types: List[str], 
                       k: int = 5) -> List[Tuple[Document, float]]:
        """Search with document type filtering."""
        results = self.search(query, k * 2)
        filtered = [(doc, score) for doc, score in results if doc.doc_type in doc_types]
        return filtered[:k]
    
    def get_document_by_citation(self, citation: str) -> Optional[Document]:
        """Retrieve a specific document by its citation."""
        for doc in self.documents:
            if doc.citation == citation:
                return doc
        return None
    
    def clear_cache(self):
        """Clear all caches."""
        self._query_cache.clear()
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)
