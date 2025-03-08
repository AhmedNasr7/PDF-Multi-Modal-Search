import re
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List 
from flair.models import SequenceTagger
from flair.data import Sentence
from flair.splitter import SegtokSentenceSplitter
from typing import Dict, List
# nltk.download('punkt')


class TextProcessor:
    """
    Handles text processing, chunking[optional], and embedding generation.
    """

    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """
        Initializes text processor with an embedding model.
        :param embedding_model: Name of the SentenceTransformer model.
        """
        self.model = SentenceTransformer(embedding_model)
        self.splitter = SegtokSentenceSplitter()

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

    def extract_text_items(self, structured_data: Dict):
        content = structured_data["content"]
        items = []
        for item in content:
            if item["type"] == "text":
                items.append(item["text"])
            elif item["type"] == "image":
                items.append(item.get("caption", ""))
        
        return items

            


    def chunk_text(self, text, chunk_size=500, chunk_overlap=50, method="recursive"):
        """
        Splits text into smaller, overlapping chunks.
        Supports LangChain's method and custom NLTK chunking.
        """

   

        if method == "recursive":
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            return splitter.split_text(text)

        elif method == "semantic":
            return self.semantic_splitter(text)
            
        # elif method == "nltk":
        #     sentences = nltk.sent_tokenize(text)
        #     chunks = []
        #     current_chunk, current_length = [], 0
        #     for sentence in sentences:
        #         if current_length + len(sentence) <= chunk_size:
        #             current_chunk.append(sentence)
        #             current_length += len(sentence)
        #         else:
        #             chunks.append(" ".join(current_chunk))
        #             current_chunk = [sentence]
        #             current_length = len(sentence)
        #     if current_chunk:
        #         chunks.append(" ".join(current_chunk))
        #     return chunks


    def semantic_splitter(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 10) -> List[str]:
     
        
        
        # Split text into sentences
        sentences = self.splitter.split(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Add sentence to the current chunk
            if len(current_chunk) + len(sentence.to_plain_string()) <= chunk_size:
                current_chunk += " " + sentence.to_plain_string()
            else:
                # If adding the next sentence exceeds max size, start a new chunk
                chunks.append(current_chunk.strip())
                current_chunk = sentence.to_plain_string()

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


    def embed_chunks(self, chunks):
        """
        Converts text chunks into vector embeddings.
        :param chunks: List of text chunks.
        :return: List of embeddings.
        """
        return self.model.encode(chunks, convert_to_tensor=False).tolist()
