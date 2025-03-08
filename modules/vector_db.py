import qdrant_client
import numpy as np
from qdrant_client.models import VectorParams, PointStruct

class VectorDBHandler:
    """
    Handles interaction with Qdrant (Vector Database).
    """

    def __init__(self, collection_name="pdf_queries"):
        """
        Initializes Qdrant vector storage.
        :param collection_name: Name of the Qdrant collection.
        """
        self.client = qdrant_client.QdrantClient(":memory:")  # Use local storage
        self.collection_name = collection_name

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance="Cosine") # cosine similarity and 
        )

    def store_vectors(self, chunks, embeddings):
        """
        Stores text chunks in Qdrant with embeddings.
        :param chunks: List of text chunks.
        :param embeddings: Corresponding embeddings.
        """
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=np.random.randint(1e6),
                        vector=vector,
                        payload={"text": chunk}
                    )
                ]
            )
    def search_vectors(self, query_text, text_processor, top_k=5):
        """
        Searches Qdrant for multiple relevant text chunks and ranks them by highest cosine similarity.
        
        :param query_text: User query.
        :param text_processor: Instance of TextProcessor.
        :param top_k: Number of retrieved documents.
        :return: List of retrieved text chunks, sorted by highest similarity.
        """
        query_vector = text_processor.model.encode(query_text).tolist()
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Sort results based on similarity score (descending order)
        ranked_results = sorted(results, key=lambda hit: hit.score, reverse=True)

        return [hit.payload["text"] for hit in ranked_results]

    # def search_multi_chunks(self, query_text, text_processor, top_k=5):
    #     """
    #     Searches Qdrant for multiple relevant text chunks.
    #     :param query_text: User query.
    #     :param text_processor: Instance of TextProcessor.
    #     :param top_k: Number of retrieved documents.
    #     :return: List of retrieved text chunks.
    #     """
    #     query_vector = text_processor.model.encode(query_text).tolist()
    #     results = self.client.search(
    #         collection_name=self.collection_name,
    #         query_vector=query_vector,
    #         limit=top_k
    #     )
    #     return [hit.payload["text"] for hit in results]
