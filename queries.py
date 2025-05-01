"""
queries.py - Test queries for evaluating LLM and RAG systems
This file contains various sets of queries designed to test and compare different RAG implementations:
- Basic LLM
- Self-RAG
- Fusion RAG
- CRAG

Each set is provided in both English and Portuguese (Portugal) for multilingual testing.
"""

# Factual retrieval queries - tests ability to retrieve specific facts accurately
factual_queries_en = [
    "What are the primary differences between transformer and recurrent neural networks?",
    "How does PageRank algorithm work and what are its limitations?",
    "What is the difference between L1 and L2 regularization in machine learning?",
    "Explain how ROUGE and BLEU metrics evaluate text generation quality",
    "What are the key components of BERT's architecture?",
    "How does beam search differ from greedy decoding in sequence generation?",
    "What is the mathematical formulation of attention mechanism in transformers?",
    "Explain the differences between precision and recall in information retrieval",
    "What are knowledge distillation techniques in deep learning?",
    "How does few-shot learning differ from zero-shot learning in NLP?"
]

factual_queries_pt = [
    "Quais são as principais diferenças entre redes neuronais transformers e recorrentes?",
    "Como funciona o algoritmo PageRank e quais são as suas limitações?",
    "Qual é a diferença entre regularização L1 e L2 em aprendizagem automática?",
    "Explique como as métricas ROUGE e BLEU avaliam a qualidade da geração de texto",
    "Quais são os componentes principais da arquitetura BERT?",
    "Como é que a pesquisa em feixe (beam search) difere da descodificação gulosa na geração de sequências?",
    "Qual é a formulação matemática do mecanismo de atenção nos transformers?",
    "Explique as diferenças entre precisão e recall na recuperação de informação",
    "O que são técnicas de destilação de conhecimento em deep learning?",
    "Como é que a aprendizagem few-shot difere da aprendizagem zero-shot em PNL?"
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

reasoning_queries_pt = [
    "Como é que os sistemas RAG lidam com informações contraditórias de diferentes fontes?",
    "Quais são as implicações éticas de utilizar sistemas RAG que possam alucinar factos?",
    "Compare e contraste a eficiência de diferentes métodos de indexação para bases de dados vetoriais",
    "Como é que os modelos de embeddings podem introduzir enviesamentos nos sistemas RAG?",
    "Que abordagens poderiam resolver o problema da informação desatualizada em sistemas RAG?",
    "Como podemos avaliar se um sistema RAG realmente compreende relações semânticas?",
    "Quais são os compromissos computacionais entre a recuperação ao nível do token e ao nível da passagem?",
    "Como poderão as arquiteturas RAG evoluir para lidar com consultas multimodais?",
    "Que técnicas poderiam melhorar o desempenho do RAG em línguas minoritárias com dados limitados?",
    "Como podemos medir e minimizar alucinações em sistemas RAG?"
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

ambiguous_queries_pt = [
    "Quais são as melhores abordagens para bases de dados vetoriais?",
    "Como funciona a recuperação?",
    "Pode explicar embeddings?",
    "O que é melhor para sistemas RAG?",
    "Como avaliar as saídas de LLM?",
    "Fale-me sobre janelas de contexto",
    "O que devo considerar ao construir um sistema RAG?",
    "Como é que os transformers lidam com documentos longos?",
    "Qual arquitetura de modelo é superior?",
    "Quais são as limitações das abordagens atuais?"
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

complex_queries_pt = [
    "Compare as implicações de desempenho de utilizar recuperadores esparsos vs densos, e explique quando escolheria um em detrimento do outro em sistemas RAG de produção",
    "Explique como diferentes estratégias de segmentação afetam o desempenho do RAG, fornecendo exemplos específicos de quando a segmentação ao nível da frase, do parágrafo e semântica seria mais apropriada",
    "Analise o impacto da engenharia de prompts na qualidade das saídas RAG, e forneça exemplos específicos de técnicas de prompt eficazes para diferentes tipos de tarefas de recuperação",
    "Discuta as diferenças arquitetónicas entre as implementações Self-RAG, Fusion RAG e CRAG, incluindo as suas vantagens únicas e potenciais modos de falha",
    "Explique como a seleção do modelo de embedding impacta o desempenho RAG em diferentes domínios, e recomende modelos de embedding específicos para documentação técnica, textos jurídicos e dados conversacionais",
    "Compare estratégias de reclassificação para sistemas RAG e explique como a fusão de classificação recíproca difere da reclassificação de codificador cruzado em termos de sobrecarga computacional e eficácia",
    "Descreva métodos para avaliar alucinações em sistemas RAG, incluindo métricas automatizadas e abordagens de avaliação humana, com os seus respetivos pontos fortes e limitações",
    "Analise como técnicas de expansão de consultas e métodos de pesquisa híbrida podem melhorar o desempenho RAG para consultas com baixa sobreposição lexical com o corpus",
    "Explique o papel do fine-tuning na melhoria do desempenho do sistema RAG, incluindo quais componentes beneficiam mais do fine-tuning e quais conjuntos de dados seriam mais apropriados",
    "Discuta estratégias para lidar com bases de conhecimento em evolução em sistemas RAG, incluindo abordagens eficientes de reindexação e mecanismos de atualização incremental"
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

technical_queries_pt = [
    "Explique como funciona a indexação HNSW em bases de dados vetoriais e as suas características de desempenho em comparação com índices planos",
    "Quais são as limitações do BM25 para pesquisa semântica e como é que os recuperadores híbridos abordam estas limitações?",
    "Como é que o mecanismo de atenção lida com o problema de complexidade quadrática nos transformers?",
    "Descreva as diferenças entre bi-encoders e cross-encoders em sistemas de recuperação de informação",
    "Como é que abordagens de aprendizagem contrastiva como SimCSE melhoram a qualidade dos embeddings para recuperação?",
    "Quais são os estrangulamentos computacionais na implementação de RLHF para sistemas RAG?",
    "Explique como a quantização afeta o desempenho e a precisão dos modelos de embedding na recuperação",
    "Quais são as abordagens atuais para lidar com raciocínio temporal em sistemas RAG?",
    "Como é que as arquiteturas de recuperação de interação tardia como ColBERT diferem da recuperação densa tradicional?",
    "Que técnicas são utilizadas para mitigar o esquecimento catastrófico em modelos de linguagem aumentados por recuperação?"
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

edge_case_queries_pt = [
    "Explique a computação quântica utilizando apenas termos da aprendizagem automática clássica",
    "Quais são os modos de falha mais comuns em RAG que não envolvem alucinação?",
    "Como desenharia um sistema RAG para recuperar informações de partituras musicais?",
    "Compare abordagens RAG para repositórios de código versus documentos em linguagem natural",
    "Como podem os sistemas RAG lidar eficazmente com negação e contrafactuais?",
    "Que técnicas utilizaria para RAG em fórmulas e equações matemáticas?",
    "Como devem os sistemas RAG lidar com consultas que exigem raciocínio numérico?",
    "Quais são os desafios da implementação de RAG para literatura médica altamente especializada?",
    "Como desenharia um sistema RAG para recuperar informações de um corpus em múltiplas línguas?",
    "Que abordagens poderiam tornar os sistemas RAG mais robustos contra consultas adversárias?"
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

multihop_queries_pt = [
    "Quais são as conexões entre técnicas de redução de dimensionalidade e desempenho de recuperação em sistemas RAG?",
    "Como é que os princípios da teoria da informação se relacionam com estratégias ótimas de segmentação para recuperação de documentos?",
    "Compare como diferentes arquiteturas RAG lidam com o compromisso entre relevância e diversidade do contexto",
    "Explique como os conceitos de transfer learning se aplicam à geração aumentada por recuperação entre domínios",
    "Que métodos combinam estruturas de grafos de conhecimento com embeddings vetoriais para melhorar a recuperação?",
    "Como é que diferentes abordagens de citação e atribuição nas saídas RAG afetam a confiança e fiabilidade do utilizador?",
    "Que técnicas fazem a ponte entre o raciocínio simbólico e a recuperação neural em sistemas RAG modernos?",
    "Como podem os princípios de aprendizagem ativa melhorar os sistemas RAG quando lidam com adaptação de domínio?",
    "Explique como os conceitos de inferência causal podem melhorar a explicabilidade das saídas RAG",
    "Quais são as conexões entre métricas tradicionais de avaliação de recuperação de informação e medidas modernas de desempenho RAG?"
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

optimization_queries_pt = [
    "Que técnicas podem reduzir o uso de tokens mantendo a qualidade do RAG?",
    "Como pode o pré-processamento de consultas melhorar a eficiência da recuperação em sistemas RAG?",
    "Quais são os compromissos entre precisão e latência em diferentes arquiteturas RAG?",
    "Compare estratégias de caching para sistemas RAG de alto rendimento",
    "Como podem os sistemas RAG ser otimizados para servir com recursos computacionais limitados?",
    "Quais são as abordagens efetivas de processamento em lote para sistemas RAG que lidam com múltiplas consultas?",
    "Como se comparam diferentes implementações de bases de dados vetoriais em termos de rendimento e latência de consulta?",
    "Que técnicas podem reduzir a pegada de memória de grandes índices de embedding?",
    "Como podem os algoritmos de vizinhos mais próximos aproximados ser afinados para desempenho RAG ideal?",
    "Quais são abordagens eficazes para paralelização de pipelines RAG?"
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

all_query_sets_pt = {
    "factual": factual_queries_pt,
    "reasoning": reasoning_queries_pt,
    "ambiguous": ambiguous_queries_pt,
    "complex": complex_queries_pt,
    "technical": technical_queries_pt,
    "edge_case": edge_case_queries_pt,
    "multihop": multihop_queries_pt,
    "optimization": optimization_queries_pt
}

# Helper function to get a specific query set in either language
def get_query_set(query_type, language="en"):
    """
    Retrieves a specific query set in the specified language.
    
    Args:
        query_type (str): Type of query set (factual, reasoning, ambiguous, etc.)
        language (str): Language code - "en" for English, "pt" for Portuguese
        
    Returns:
        list: List of queries in the specified language
    """
    if language.lower() == "en":
        return all_query_sets_en.get(query_type.lower(), [])
    elif language.lower() == "pt":
        return all_query_sets_pt.get(query_type.lower(), [])
    else:
        raise ValueError(f"Unsupported language: {language}. Use 'en' or 'pt'.")

# Example usage:
# factual_en = get_query_set("factual", "en")
# factual_pt = get_query_set("factual", "pt")