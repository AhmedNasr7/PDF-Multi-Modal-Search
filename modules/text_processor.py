import nltk
import re
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

nltk.download('punkt')

class TextProcessor:
    """
    Handles text processing, chunking, and embedding generation.
    """

    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """
        Initializes text processor with an embedding model.
        :param embedding_model: Name of the SentenceTransformer model.
        """
        self.model = SentenceTransformer(embedding_model)

    def build_text_document(self, structured_data):
        """
        Combines all text and image captions into a single ordered document.
        :param structured_data: The parsed document JSON.
        :return: A merged text document.
        """
        document_text = []
        for item in structured_data["content"]:
            if item["type"] == "text":
                document_text.append(item["text"])
            elif item["type"] == "image":
                caption = item["caption"] if item["caption"] else f"[Image {item['index']}]"
                document_text.append(f"Image Caption: {caption}")
        return "\n\n".join(document_text)

    def chunk_text(self, text, chunk_size=300, chunk_overlap=50, method="nltk"):
        """
        Splits text into smaller, overlapping chunks.
        Supports LangChain's method and custom NLTK chunking.
        """
        if method == "langchain":
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            return splitter.split_text(text)

        elif method == "nltk":
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk, current_length = [], 0
            for sentence in sentences:
                if current_length + len(sentence) <= chunk_size:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = len(sentence)
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks

    def embed_chunks(self, chunks):
        """
        Converts text chunks into vector embeddings.
        :param chunks: List of text chunks.
        :return: List of embeddings.
        """
        return self.model.encode(chunks, convert_to_tensor=False).tolist()
