import streamlit as st
import os
import tempfile
from typing import List
from modules.document_parser import DocumentParser
from modules.text_processor import TextProcessor
from modules.vector_db import VectorDBHandler
from modules.reranker import ReRanker
from modules.answer_merger import T5AnswerMerger
from modules.query_pipeline import QueryPipeline
from modules.vlm_service import VLMService

# Streamlit page setup
st.set_page_config(page_title="📄 AI PDF Query System", layout="wide")
st.title("📄 AI PDF Query System")

# Initialize components
API_URL = "http://localhost:8000/caption_image"

vlm_service = VLMService(api_url=API_URL)
parser = DocumentParser(vlm_service)
text_processor = TextProcessor()
vector_db = VectorDBHandler()
reranker = None
t5_merger = None

pipeline = QueryPipeline(text_processor, vector_db, reranker, t5_merger)

# File uploader
uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        pdf_path = temp_file.name

    st.success("✅ PDF uploaded successfully! Processing now...")

    # Parse PDF
    with st.spinner("🔄 Extracting text and images..."):
        structured_data = parser.parse_pdf(pdf_path)
        text_chunks = text_processor.extract_text_items(structured_data)
        embeddings = text_processor.embed_chunks(text_chunks)
        vector_db.store_vectors(text_chunks, embeddings)

    # Delete temp file after processing
    os.remove(pdf_path)

    st.success("✅ Document processed and stored in Qdrant!")
    
    # Store structured data in session state
    st.session_state["structured_data"] = structured_data
    st.session_state["document_uploaded"] = True

# Query section
st.header("🔎 Ask a Question About the PDF")
query = st.text_input("Enter your question:")

if st.button("Search"):
    if not uploaded_file:
        st.error("⚠️ Please upload a PDF first!")
    elif not query:
        st.error("⚠️ Please enter a question!")
    else:
        with st.spinner("🔄 Searching for the best answer..."):
            answer = pipeline.process_query(query)

        # Handle empty results
        if isinstance(answer, List) and answer:
            answer = answer[0]  # Take the top answer
        else:
            answer = "No relevant information found."

        st.success("✅ Answer retrieved!")
        st.write("🔹 **Answer:**")
        st.write(answer)

# Run Streamlit app with: `streamlit run app.py`
