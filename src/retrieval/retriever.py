"""
Retrieval module with caching and advanced cognitive retrieval.
Implements:
- Parent-Document retrieval resolution.
- Hybrid sparse-dense search using Reciprocal Rank Fusion (RRF).
- Gemini-powered document reranking.
"""
import math
import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from ..config import settings
from ..processing import VectorStorePipeline
from .parent_retriever import ParentDocumentStore


class SimpleBM25:
    """A lightweight, native Python BM25 implementation for lexical candidate scoring."""
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc.split()) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.doc_freqs = []
        self.idf = {}
        
        # Calculate term frequencies and document frequencies
        for doc in corpus:
            freq = {}
            for word in doc.lower().split():
                # Clean punctuation
                word = ''.join(c for c in word if c.isalnum())
                if word:
                    freq[word] = freq.get(word, 0) + 1
            self.doc_freqs.append(freq)
            
            for word in freq:
                self.idf[word] = self.idf.get(word, 0) + 1
                
        # Calculate IDF
        for word, df in self.idf.items():
            self.idf[word] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str, index: int) -> float:
        """Score a query against a specific document index."""
        score = 0.0
        doc_freq = self.doc_freqs[index]
        doc_len = self.doc_lengths[index]
        for word in query.lower().split():
            word = ''.join(c for c in word if c.isalnum())
            if word in doc_freq:
                tf = doc_freq[word]
                idf = self.idf.get(word, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
        return score


class CachedRetriever:
    """Retriever with configurable parameters, caching, and advanced search pipelines."""
    
    def __init__(
        self,
        vectorstore: Optional[VectorStore] = None,
        k: int = None,
        score_threshold: float = None,
        search_type: str = "similarity"
    ):
        self.k = k or settings.retrieval_top_k
        self.score_threshold = score_threshold or settings.retrieval_score_threshold
        self.search_type = search_type
        
        # Initialize vector store if not provided
        if vectorstore is None:
            vector_pipeline = VectorStorePipeline()
            self.vectorstore = vector_pipeline.initialize_vectorstore()
        else:
            self.vectorstore = vectorstore
        
        # Create LangChain retriever
        self.retriever = self._create_retriever()
        self.parent_store = ParentDocumentStore()
    
    def _create_retriever(self) -> BaseRetriever:
        """Create LangChain retriever with configured parameters."""
        search_kwargs = {
            "k": self.k,
        }
        if self.search_type == "similarity":
            search_kwargs["score_threshold"] = self.score_threshold
        
        return self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query, running hybrid search,
        parent document lookup, and reranking if enabled.
        """
        # Step 1: Execute Dense/Hybrid Retrieval
        if settings.enable_hybrid_search:
            print("[RETRIEVE] Executing Hybrid sparse-dense search (RRF)")
            # Fetch a larger candidate pool for fusion
            candidate_k = self.k * 2
            results_with_scores = self.vectorstore.similarity_search_with_score(query, k=candidate_k)
            dense_docs = [doc for doc, _ in results_with_scores]
            
            if not dense_docs:
                documents = []
            else:
                # Rank candidates with BM25
                corpus = [doc.page_content for doc in dense_docs]
                bm25 = SimpleBM25(corpus)
                scored_indices = sorted(
                    range(len(dense_docs)),
                    key=lambda idx: bm25.score(query, idx),
                    reverse=True
                )
                sparse_docs = [dense_docs[idx] for idx in scored_indices]
                
                # Perform Reciprocal Rank Fusion
                documents = self._reciprocal_rank_fusion(dense_docs, sparse_docs)[:self.k]
        else:
            print("[RETRIEVE] Executing Standard vector search")
            documents = self.retriever.invoke(query)
            
        # Step 2: Resolve Parent Documents if enabled
        if settings.enable_parent_document_retrieval:
            print("[RETRIEVE] Resolving Parent Documents")
            resolved_docs = []
            seen_parent_ids = set()
            for doc in documents:
                parent_id = doc.metadata.get("parent_id")
                if parent_id:
                    if parent_id not in seen_parent_ids:
                        parent_doc = self.parent_store.get(parent_id)
                        if parent_doc:
                            resolved_docs.append(parent_doc)
                            seen_parent_ids.add(parent_id)
                        else:
                            resolved_docs.append(doc)
                else:
                    resolved_docs.append(doc)
            documents = resolved_docs
            
        # Step 3: Run Reranking if enabled
        if settings.enable_reranking:
            print("[RETRIEVE] Running Gemini Reranking")
            documents = self._gemini_rerank(query, documents, top_n=self.k)
            
        print(f"[RETRIEVE] Retrieved {len(documents)} documents for query")
        return documents
    
    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """Retrieve documents with relevance scores."""
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        
        # Filter by score threshold
        filtered_results = [
            (doc, score) for doc, score in results
            if score >= self.score_threshold
        ]
        
        print(f"Retrieved {len(filtered_results)} documents (filtered from {len(results)})")
        return filtered_results
    
    def _reciprocal_rank_fusion(self, dense_docs: List[Document], sparse_docs: List[Document], k: int = 60) -> List[Document]:
        """Combine dense and sparse search rankings using RRF."""
        rrf_scores = {}
        
        # Score dense rank
        for rank, doc in enumerate(dense_docs):
            doc_key = doc.page_content
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)
            
        # Score sparse rank
        for rank, doc in enumerate(sparse_docs):
            doc_key = doc.page_content
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)
            
        # Reconstruct and sort
        all_docs = {doc.page_content: doc for doc in dense_docs + sparse_docs}
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [all_docs[key] for key in sorted_keys]
        
    def _gemini_rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """Use Gemini to rerank documents by relevance."""
        if not documents:
            return []
            
        from ..llm import get_gemini_client
        llm_client = get_gemini_client()
        llm = llm_client.chat_model
        
        doc_list = []
        for idx, doc in enumerate(documents):
            # Send first 500 chars to save tokens
            doc_list.append(f"ID: {idx}\nContent: {doc.page_content[:500]}")
            
        prompt = f"""You are a search reranker. Grade and rank the relevance of the following documents to the query.
Query: {query}

Documents:
{"\n---\n".join(doc_list)}

Order the documents by relevance. Return ONLY a JSON list of document IDs (integers) in order of most relevant to least relevant. Do not explain your reasoning.
Example response: [2, 0, 1]
JSON Response:"""
        
        try:
            response = llm.invoke(prompt).content
            match = re.search(r"\[\s*\d+\s*(?:,\s*\d+\s*)*\]", response)
            if match:
                order = json.loads(match.group(0))
                reranked_docs = []
                seen = set()
                for idx in order:
                    if 0 <= idx < len(documents) and idx not in seen:
                        reranked_docs.append(documents[idx])
                        seen.add(idx)
                # Append any docs that were missed
                for idx, doc in enumerate(documents):
                    if idx not in seen:
                        reranked_docs.append(doc)
                return reranked_docs[:top_n]
        except Exception as e:
            print(f"[RERANK] Error in Gemini Reranking: {e}. Returning original candidates.")
            
        return documents[:top_n]

    def get_document_ids(self, documents: List[Document]) -> List[str]:
        """Extract unique identifiers from documents for cache keys."""
        doc_ids = []
        for doc in documents:
            doc_id = doc.metadata.get("id") or doc.metadata.get("source", "unknown")
            if "page" in doc.metadata:
                doc_id = f"{doc_id}_page_{doc.metadata['page']}"
            doc_ids.append(str(hash(doc_id)))
        return doc_ids
    
    def format_documents(self, documents: List[Document]) -> str:
        """Format documents into a single context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""
            
            context_parts.append(
                f"[Document {i}] Source: {source}{page_info}\n"
                f"{doc.page_content}\n"
            )
        return "\n---\n".join(context_parts)
    
    def get_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents."""
        sources = []
        for doc in documents:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "type": doc.metadata.get("type", "Unknown"),
            }
            if "page" in doc.metadata:
                source_info["page"] = doc.metadata["page"]
            if "filename" in doc.metadata:
                source_info["filename"] = doc.metadata["filename"]
            if "url" in doc.metadata:
                source_info["url"] = doc.metadata["url"]
            sources.append(source_info)
        return sources


class RetrievalManager:
    """Manages retrieval operations with multiple retriever configurations."""
    
    def __init__(self):
        self.default_retriever = CachedRetriever()
        self.custom_retrievers: Dict[str, CachedRetriever] = {}
    
    def create_retriever(
        self,
        name: str,
        k: int = None,
        score_threshold: float = None,
        search_type: str = "similarity"
    ) -> CachedRetriever:
        """Create and register a custom retriever."""
        retriever = CachedRetriever(
            k=k,
            score_threshold=score_threshold,
            search_type=search_type
        )
        self.custom_retrievers[name] = retriever
        return retriever
    
    def get_retriever(self, name: Optional[str] = None) -> CachedRetriever:
        """Get a retriever by name or return default."""
        if name and name in self.custom_retrievers:
            return self.custom_retrievers[name]
        return self.default_retriever
    
    def retrieve(
        self,
        query: str,
        retriever_name: Optional[str] = None
    ) -> List[Document]:
        """Retrieve documents using specified or default retriever."""
        retriever = self.get_retriever(retriever_name)
        return retriever.retrieve(query)


def create_retriever(
    k: int = None,
    score_threshold: float = None
) -> CachedRetriever:
    """Factory function to create a retriever."""
    return CachedRetriever(k=k, score_threshold=score_threshold)
