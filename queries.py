"""
queries.py - Test queries for evaluating LLM and RAG systems
This file contains various sets of queries designed to test and compare different RAG implementations:
- Basic LLM
- Self-RAG
- Fusion RAG
- CRAG

"""

# Factual retrieval queries - tests ability to retrieve specific facts accurately
factual_queries_en = [
    (
        "What are the primary differences between transformer and recurrent neural networks?",
        "Transformers process all tokens in a sequence simultaneously using self-attention mechanisms, allowing for parallelization and better long-range dependency modeling. Recurrent Neural Networks (RNNs), including LSTMs and GRUs, process input sequentially, which limits parallelism and makes capturing long-term dependencies more difficult. Transformers also use positional encoding to maintain order, while RNNs inherently maintain sequence order through recurrence."
    ),
    (
        "How does PageRank algorithm work and what are its limitations?",
        "PageRank is an algorithm that assigns a numerical weight to each webpage based on its importance, calculated iteratively using the link structure of the web. It models web navigation as a Markov chain, where the rank of a page is determined by the ranks of the pages linking to it. Limitations include susceptibility to link spam, inability to capture semantic relevance, and issues with ranking new or low-link pages. It also assumes equal likelihood of link-following, which may not reflect user behavior."
    ),
    #(
    #    "What is the difference between L1 and L2 regularization in machine learning?",
    #    "L1 regularization (Lasso) adds the absolute value of coefficients to the loss function, promoting sparsity and feature selection. L2 regularization (Ridge) adds the squared values of coefficients, encouraging smaller but non-zero weights. L1 can drive weights to zero, effectively removing features, while L2 generally shrinks weights without making them exactly zero. The choice between them depends on the problem, such as interpretability vs. smoothness."
    #),
    #(
    #    "Explain how ROUGE and BLEU metrics evaluate text generation quality",
    #    "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) evaluates text by measuring overlap of n-grams, word sequences, and word pairs between generated and reference texts, focusing on recall. BLEU (Bilingual Evaluation Understudy) measures the precision of n-gram overlaps between the generated text and references, typically used in machine translation. BLEU penalizes short outputs through brevity penalty, while ROUGE is better for summarization where recall is more important."
    #),
    #(
    #    "What are the key components of BERT's architecture?",
    #    "BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder stack of the original transformer architecture. Key components include multi-head self-attention layers, feed-forward neural networks, layer normalization, and residual connections. It is pre-trained using masked language modeling (MLM) and next sentence prediction (NSP) tasks. BERT is bidirectional, meaning it attends to both left and right context simultaneously, enabling deeper understanding of word meaning."
    #),
    #(
    #    "How does beam search differ from greedy decoding in sequence generation?",
    #    "Greedy decoding selects the highest probability token at each step, which can lead to suboptimal overall sequences. Beam search maintains multiple hypotheses (beams) at each time step, expanding and keeping the top-k sequences with the highest cumulative probability. This allows it to explore a larger space and typically results in more coherent and accurate outputs, though it is more computationally expensive than greedy decoding."
    #),
    #(
    #    "What is the mathematical formulation of attention mechanism in transformers?",
    #    "The scaled dot-product attention is formulated as: Attention(Q, K, V) = softmax(QK^T / √d_k) V, where Q (query), K (key), and V (value) are linear projections of the input embeddings, and d_k is the dimensionality of the key vectors. This computes attention weights by measuring the similarity between queries and keys, then applies them to the values to generate context-aware representations."
    #),
    #(
    #    "Explain the differences between precision and recall in information retrieval",
    #    "Precision is the ratio of relevant documents retrieved to the total retrieved (true positives / (true positives + false positives)), measuring accuracy. Recall is the ratio of relevant documents retrieved to the total relevant documents available (true positives / (true positives + false negatives)), measuring completeness. High precision means fewer irrelevant results; high recall means most relevant results are retrieved. They often trade off against each other."
    #),
    #(
    #    "What are knowledge distillation techniques in deep learning?",
    #    "Knowledge distillation involves training a smaller, more efficient student model to mimic the behavior of a larger, more complex teacher model. This is done by minimizing the difference between the student’s outputs and the soft targets (logits) produced by the teacher. Techniques include soft-label distillation, intermediate representation matching, and response-based or feature-based transfer. It enables model compression while retaining much of the performance of the original model."
    #),
    #(
    #    "How does few-shot learning differ from zero-shot learning in NLP?",
    #    "Few-shot learning involves training or adapting a model with a small number of labeled examples per class, while zero-shot learning requires the model to generalize to tasks or classes it has never seen before, based on semantic information or task descriptions. Zero-shot typically leverages pretrained models with natural language prompts or external knowledge. Few-shot relies on in-context learning or fine-tuning with minimal data."
    #)
]

# Reasoning queries - tests logical reasoning and inference capability
reasoning_queries_en = [
    "How would RAG systems handle contradictory information from different sources?",
    "What are the ethical implications of using RAG systems that might hallucinate facts?",
    "Compare and contrast the efficiency of different indexing methods for vector databases",
    "How might embedding models introduce bias in RAG systems?",
    "What approaches could solve the problem of outdated information in RAG systems?",
    "How can we evaluate whether a RAG system truly understands semantic relationships?",
    "What are the computational trade-offs between token-level and passage-level retrieval?",
    "How might RAG architectures evolve to handle multimodal queries?",
    "What techniques could improve RAG performance on minority languages with limited data?",
    "How can we measure and minimize hallucinations in RAG systems?"
]

# Ambiguous queries - tests handling of vagueness and clarification capability
ambiguous_queries_en = [
    "What are the best approaches for vector databases?",
    "How does retrieval work?",
    "Can you explain embeddings?",
    "What's better for RAG systems?",
    "How to evaluate LLM outputs?",
    "Tell me about context windows",
    "What should I consider when building a RAG system?",
    "How do transformers handle long documents?",
    "Which model architecture is superior?",
    "What are the limitations of current approaches?"
]

# Complex queries - test ability to handle multi-part questions and synthesize information
complex_queries_en = [
    "Compare the performance implications of using sparse vs dense retrievers, and explain when you would choose one over the other in production RAG systems",
    "Explain how different chunking strategies affect RAG performance, providing specific examples of when sentence-level, paragraph-level, and semantic chunking would be most appropriate",
    "Analyze the impact of prompt engineering on the quality of RAG outputs, and provide specific examples of effective prompting techniques for different types of retrieval tasks",
    "Discuss the architectural differences between Self-RAG, Fusion RAG, and CRAG implementations, including their unique advantages and potential failure modes",
    "Explain how embedding model selection impacts RAG performance across different domains, and recommend specific embedding models for technical documentation, legal texts, and conversational data",
    "Compare reranking strategies for RAG systems and explain how reciprocal rank fusion differs from cross-encoder reranking in terms of computational overhead and effectiveness",
    "Describe methods for evaluating hallucination in RAG systems, including automated metrics and human evaluation approaches, with their respective strengths and limitations",
    "Analyze how query expansion techniques and hybrid search methods can improve RAG performance for queries with low lexical overlap with the corpus",
    "Explain the role of fine-tuning in improving RAG system performance, including which components benefit most from fine-tuning and what datasets would be most appropriate",
    "Discuss strategies for handling evolving knowledge bases in RAG systems, including efficient reindexing approaches and incremental updating mechanisms"
]

# Technical specific queries - test domain expertise and technical depth
technical_queries_en = [
    "Explain how HNSW indexing works in vector databases and its performance characteristics compared to flat indices",
    "What are the limitations of BM25 for semantic search and how do hybrid retrievers address these limitations?",
    "How does the attention mechanism deal with the quadratic complexity problem in transformers?",
    "Describe the differences between bi-encoders and cross-encoders in information retrieval systems",
    "How do contrastive learning approaches like SimCSE improve embedding quality for retrieval?",
    "What are the computational bottlenecks in implementing RLHF for RAG systems?",
    "Explain how quantization affects the performance and accuracy of embedding models in retrieval",
    "What are the current approaches to handle temporal reasoning in RAG systems?",
    "How do late-interaction retrieval architectures like ColBERT differ from traditional dense retrieval?",
    "What techniques are used to mitigate catastrophic forgetting in retrieval-augmented language models?"
]

# Edge case queries - test robustness to unusual or challenging queries
edge_case_queries_en = [
    "Explain quantum computing using only terms from classical machine learning",
    "What are the most common RAG failure modes that don't involve hallucination?",
    "How would you design a RAG system for retrieving information from sheet music?",
    "Compare RAG approaches for code repositories versus natural language documents",
    "How can RAG systems effectively handle negation and counterfactuals?",
    "What techniques would you use for RAG on mathematical formulas and equations?",
    "How should RAG systems handle queries requiring numerical reasoning?",
    "What are the challenges of implementing RAG for highly specialized medical literature?",
    "How would you design a RAG system for retrieving information from a corpus in multiple languages?",
    "What approaches could make RAG systems more robust to adversarial queries?"
]

# Multi-hop queries - test ability to connect information across multiple documents
multihop_queries_en = [
    "What are the connections between dimensionality reduction techniques and retrieval performance in RAG systems?",
    "How do the principles of information theory relate to optimal chunking strategies for document retrieval?",
    "Compare how different RAG architectures handle the trade-off between context relevance and diversity",
    "Explain how transfer learning concepts apply to cross-domain retrieval augmented generation",
    "What methods combine knowledge graph structures with vector embeddings for improved retrieval?",
    "How do different citation and attribution approaches in RAG outputs affect user trust and reliability?",
    "What techniques bridge the gap between symbolic reasoning and neural retrieval in modern RAG systems?",
    "How can active learning principles improve RAG systems when dealing with domain adaptation?",
    "Explain how causal inference concepts can improve the explainability of RAG outputs",
    "What are the connections between traditional information retrieval evaluation metrics and modern RAG performance measures?"
]

# Performance optimization queries - test understanding of system efficiency 
optimization_queries_en = [
    "What techniques can reduce token usage while maintaining RAG quality?",
    "How can query preprocessing improve retrieval efficiency in RAG systems?",
    "What are the trade-offs between accuracy and latency in different RAG architectures?",
    "Compare caching strategies for high-throughput RAG systems",
    "How can RAG systems be optimized for serving with limited computational resources?",
    "What are effective batch processing approaches for RAG systems handling multiple queries?",
    "How do different vector database implementations compare in terms of throughput and query latency?",
    "What techniques can reduce the memory footprint of large embedding indexes?",
    "How can approximate nearest neighbor algorithms be tuned for optimal RAG performance?",
    "What are effective approaches for RAG pipeline parallelization?"
]

# All query sets combined for convenience
all_query_sets_en = {
    "factual": factual_queries_en,
    "reasoning": reasoning_queries_en,
    "ambiguous": ambiguous_queries_en,
    "complex": complex_queries_en,
    "technical": technical_queries_en,
    "edge_case": edge_case_queries_en,
    "multihop": multihop_queries_en,
    "optimization": optimization_queries_en
}

# Helper function to get a specific query set in either language
def get_query_set(query_type):
    """
    Retrieves a specific query set in the specified language.
    
    Args:
        query_type (str): Type of query set (factual, reasoning, ambiguous, etc.)
        language (str): Language code - "en" for English, "pt" for Portuguese
        
    Returns:
        list: List of queries in the specified language
    """
    return all_query_sets_en.get(query_type.lower(), [])