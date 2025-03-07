from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Optional: I might rank based on TF-IDF 

class ReRanker:
    """
    Re-ranks retrieved text snippets using TF-IDF or an LLM-based approach.
    """

    def __init__(self, method="tfidf"):
        """
        Initializes the ReRanker.
        :param method: Ranking method - "tfidf" (default) or "llm".
        """
        if method == "tfidf": # we can add "llm" rerankers 
            self.vectorizer = TfidfVectorizer()
        else:
            raise NotImplementedError(f"Re-ranking method '{method}' is not implemented yet. Use 'tfidf'.")


    def rerank_by_tfidf(self, query, results):
        """
        Re-ranks search results based on TF-IDF similarity to the query.
        :param query: The user query.
        :param results: List of (text, score) tuples from Qdrant.
        :return: Re-ranked list of text snippets.
        """
        texts = [text for text, _ in results]  # Extract only texts

        if len(texts) == 0:
            return []  # No results to rank

        
        tfidf_matrix = self.vectorizer.fit_transform([query] + texts)

        query_vector = tfidf_matrix[0]  # Query vector
        text_vectors = tfidf_matrix[1:]  # Text vectors

        # Compute cosine similarity between query and each text
        tfidf_scores = (text_vectors @ query_vector.T).toarray().flatten()

        # Sort results by TF-IDF relevance
        ranked_results = sorted(zip(texts, tfidf_scores), key=lambda x: x[1], reverse=True)
        return [text for text, _ in ranked_results]

    def rerank(self, query, results):
        """
        Calls the appropriate reranking method based on the chosen approach.
        :param query: The user query.
        :param results: List of (text, score) tuples.
        :return: Re-ranked text snippets.
        """
        return self.rerank_by_tfidf(query, results)

