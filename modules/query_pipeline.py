"""
This class implements query pipeline handler to handle all the process components & steps.
"""


class QueryPipeline:
    """
    Handles query retrieval, re-ranking, and merging of responses.
    """

    def __init__(self, text_processor, vector_db, reranker=None, t5_merger=None, ranker_method="cosine_similarity", merger_method="t5", top_k=5):
        """
        Initializes the query pipeline.
        
        :param text_processor: TextProcessor instance.
        :param vector_db: VectorDBHandler instance.
        :param reranker: ReRanker instance (Optional, depends on `ranker_method`).
        :param t5_merger: T5AnswerMerger instance (Optional, depends on `merger_method`).
        :param ranker_method: Method for ranking results ("tfidf", "cosine_similarity", "none").
        :param merger_method: Method for merging results ("t5", "concatenation").
        :param top_k: Number of retrieved text chunks.
        """
        self.text_processor = text_processor
        self.vector_db = vector_db
        self.reranker = reranker
        self.t5_merger = t5_merger
        self.ranker_method = ranker_method
        self.merger_method = merger_method
        self.top_k = top_k

    def process_query(self, query):
        """
        Retrieves, re-ranks, and summarizes answers based on the query.
        
        :param query: The user's query.
        :return: A structured response.
        """
        # query_text_chunks = self.text_processor.chunk_text(query)
        print("query: ", query, self.top_k)
        # query_embedding = self.text_processor.model.encode(query)
        # query_embedding = self.text_processor.embed_chunks([query])[0]
        retrieved_chunks = self.vector_db.search_vectors(query, self.text_processor, top_k=self.top_k)

        # Apply ranking if a method is set
        if self.ranker_method == "tfidf" and self.reranker:
            ranked_chunks = self.reranker.rerank(query, retrieved_chunks)
        elif self.ranker_method == "cosine_similarity":
            ranked_chunks = retrieved_chunks  # Already sorted by similarity from Qdrant
        else:
            ranked_chunks = retrieved_chunks  # No ranking applied

        if self.merger_method == "t5" and self.t5_merger:
            print(ranked_chunks)
            return self.t5_merger.merge_and_summarize(ranked_chunks, query)
        elif self.merger_method == "concatenation":
            return "\n".join(ranked_chunks)  # Simple concatenation of retrieved chunks
        else:
            return ranked_chunks  # Default return with the merging and reranking
