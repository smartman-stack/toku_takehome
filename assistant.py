"""
Main RAG assistant that orchestrates retrieval, generation, and policy enforcement.
Clean, modular architecture with support for LLM and offline modes.
"""
from typing import List, Dict, Any, Optional
from data_loader import DataLoader, Document
from retriever import HybridRetriever
from generator import AnswerGenerator
from policy_enforcer import PolicyEnforcer
from config import get_config, Config


class RAGAssistant:
    """
    Retrieval-augmented assistant for customer support.
    
    Features:
    - Hybrid search (semantic + BM25) with reranking
    - LLM-based answer generation with offline fallback
    - Policy enforcement (PII masking, escalation, SLA)
    - Inline citations and confidence scoring
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the assistant.
        
        Args:
            config: Optional configuration. If None, uses default/env config.
        """
        self.config = config or get_config()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.retriever: Optional[HybridRetriever] = None
        self.generator: Optional[AnswerGenerator] = None
        self.policy_enforcer = PolicyEnforcer(self.config)
        
        self._initialized = False
    
    def initialize(self):
        """Load data, create indices, and prepare all components."""
        if self._initialized:
            return
        
        print("=" * 60)
        print("Initializing RAG Assistant")
        print("=" * 60)
        
        # Load all data sources
        print("\n[1/4] Loading data sources...")
        documents = self.data_loader.load_all()
        print(f"      Loaded {len(documents)} documents from 4 sources")
        
        # Initialize retriever with hybrid search
        print("\n[2/4] Initializing hybrid retriever...")
        self.retriever = HybridRetriever(self.config)
        self.retriever.index_documents(documents)
        
        # Initialize generator
        print("\n[3/4] Initializing answer generator...")
        self.generator = AnswerGenerator(self.config)
        
        # Initialize policy enforcer (already done in __init__)
        print("\n[4/4] Policy enforcer ready")
        
        self._initialized = True
        print("\n" + "=" * 60)
        print("Assistant initialized successfully!")
        print(f"  - Retrieval: Hybrid (semantic + BM25)")
        print(f"  - Reranking: {'Enabled' if self.config.model.use_reranker else 'Disabled'}")
        print(f"  - Generation: {'LLM' if self.generator._llm_available else 'Offline'}")
        print("=" * 60 + "\n")
    
    def generate_answer(self, query: str, 
                       top_k: int = 5,
                       is_enterprise: bool = False,
                       force_offline: bool = False) -> Dict[str, Any]:
        """
        Generate answer with citations and policy enforcement.
        
        Args:
            query: Customer question
            top_k: Number of documents to retrieve
            is_enterprise: Whether customer is enterprise (affects SLA)
            force_offline: Force offline generation even if LLM available
        
        Returns:
            Dict containing:
            - answer: The generated answer (with policy enforcement applied)
            - citations: List of citations
            - escalation: Escalation information
            - sla_guidance: SLA guidance message
            - confidence: Confidence score (0-1)
            - retrieved_docs: Number of documents retrieved
            - method: Generation method used
        """
        if not self._initialized:
            self.initialize()
        
        # Step 1: Retrieve relevant documents
        retrieved = self.retriever.search(query, k=top_k)
        
        # Step 2: Generate answer
        generation_result = self.generator.generate(
            query=query,
            documents=retrieved,
            force_offline=force_offline
        )
        
        answer = generation_result["answer"]
        citations = generation_result["citations"]
        confidence = generation_result["confidence"]
        method = generation_result["method"]
        
        # Step 3: Apply policy enforcement
        policy_result = self.policy_enforcer.enforce_policies(
            text=answer,
            query=query,
            is_enterprise=is_enterprise
        )
        
        # Step 4: Build response
        final_answer = policy_result["text"]
        
        # Add escalation notice if needed
        if policy_result["escalation"]["needed"]:
            final_answer += f"\n\n⚠️ {policy_result['escalation']['message']}"
        
        return {
            "answer": final_answer,
            "citations": citations,
            "escalation": policy_result["escalation"],
            "sla_guidance": policy_result["sla_guidance"],
            "confidence": confidence,
            "retrieved_docs": len(retrieved),
            "method": method
        }
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar documents without generating an answer.
        Useful for debugging and analysis.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
        
        Returns:
            List of document info dicts
        """
        if not self._initialized:
            self.initialize()
        
        results = self.retriever.search(query, k=k)
        
        return [
            {
                "citation": doc.citation,
                "content": doc.content,
                "doc_type": doc.doc_type,
                "score": round(score, 4),
                "metadata": doc.metadata
            }
            for doc, score in results
        ]
    
    def explain_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer with detailed explanation of the process.
        Useful for debugging and understanding system behavior.
        
        Args:
            query: Customer question
        
        Returns:
            Dict with answer and detailed breakdown
        """
        if not self._initialized:
            self.initialize()
        
        # Get retrieved documents with scores
        retrieved = self.retriever.search(query, k=self.config.retrieval.initial_k)
        
        # Generate answer
        generation_result = self.generator.generate(query, retrieved)
        
        # Apply policies
        policy_result = self.policy_enforcer.enforce_policies(
            generation_result["answer"], query
        )
        
        return {
            "query": query,
            "answer": policy_result["text"],
            "breakdown": {
                "retrieved_documents": [
                    {
                        "citation": doc.citation,
                        "type": doc.doc_type,
                        "score": round(score, 4),
                        "preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                    }
                    for doc, score in retrieved
                ],
                "generation_method": generation_result["method"],
                "confidence": generation_result["confidence"],
                "escalation_triggered": policy_result["escalation"]["needed"],
                "pii_masked": policy_result["pii_masked"]
            }
        }
    
    def batch_generate(self, queries: List[str], 
                       is_enterprise: bool = False) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries.
        
        Args:
            queries: List of customer questions
            is_enterprise: Whether customer is enterprise tier
        
        Returns:
            List of response dicts
        """
        if not self._initialized:
            self.initialize()
        
        results = []
        for query in queries:
            result = self.generate_answer(query, is_enterprise=is_enterprise)
            results.append(result)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the assistant."""
        return {
            "initialized": self._initialized,
            "config": {
                "llm_provider": self.config.model.llm_provider,
                "llm_available": self.generator._llm_available if self.generator else False,
                "reranker_enabled": self.config.model.use_reranker,
                "hybrid_search": True,
                "semantic_weight": self.config.retrieval.semantic_weight,
                "bm25_weight": self.config.retrieval.bm25_weight,
            },
            "documents_indexed": len(self.retriever.documents) if self.retriever else 0
        }


# Convenience function for quick usage
def create_assistant(use_llm: bool = True, use_reranker: bool = True) -> RAGAssistant:
    """
    Factory function to create assistant with common configurations.
    
    Args:
        use_llm: Whether to enable LLM (requires API key)
        use_reranker: Whether to enable cross-encoder reranking
    
    Returns:
        Configured RAGAssistant instance
    """
    config = get_config()
    
    if not use_llm:
        config.model.llm_provider = "offline"
    
    config.model.use_reranker = use_reranker
    
    # Optimize hybrid search weights
    config.retrieval.semantic_weight = 0.6
    config.retrieval.bm25_weight = 0.4
    
    assistant = RAGAssistant(config)
    return assistant
