import argparse
import os
from modules.document_parser import DocumentParser
from modules.text_processor import TextProcessor
from modules.vector_db import VectorDBHandler
from modules.reranker import ReRanker
from modules.t5_answer_merger import T5AnswerMerger
from modules.query_pipeline import QueryPipeline
from modules.vlm_service import VLMService


API_URL = "http://localhost:8000/caption_image"

def main():
    # CLI argument parser
    parser = argparse.ArgumentParser(description="AI-Powered PDF Query System")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--ranker", type=str, choices=["tfidf", "cosine_similarity", "none"], default="cosine_similarity",
                        help="Ranking method (default: cosine_similarity)")
    parser.add_argument("--merger", type=str, choices=["t5", "concatenation"], default=None,
                        help="Merging method (default: t5)")
    parser.add_argument("--top_k", type=int, default=2, help="Number of retrieved results (default: 5)")

    args = parser.parse_args()

    # Initialize components
    print("🚀 Initializing components...")
    vlm_service = VLMService(api_url=API_URL)
    doc_parser = DocumentParser(vlm_service)
    text_processor = TextProcessor()
    vector_db = VectorDBHandler()
    reranker = ReRanker() if args.ranker == "tfidf" else None
    t5_merger = T5AnswerMerger() if args.merger == "t5" else None

    # Initialize QueryPipeline with configurations
    pipeline = QueryPipeline(
        text_processor, vector_db, reranker, t5_merger,
        ranker_method=args.ranker, merger_method=args.merger, top_k=args.top_k
    )

    # Parse PDF
    print(f"📄 Parsing PDF: {args.pdf_path}")
    if not os.path.exists(args.pdf_path):
        print("❌ Error: PDF file not found!")
        return

    structured_data = doc_parser.parse_pdf(args.pdf_path)
    
    # Process text and store in Qdrant
    document_text = " ".join([item["text"] for item in structured_data["content"] if item["type"] == "text"])
    text_chunks = text_processor.chunk_text(document_text)
    embeddings = text_processor.embed_chunks(text_chunks)
    vector_db.store_vectors(text_chunks, embeddings)

    print("✅ PDF processed and stored in Qdrant!")

    # Interactive query mode
    print("\n🔎 Enter queries below (type 'exit' to quit):")
    while True:
        query = input("\n📝 Your question: ")
        if query.lower() == "exit":
            break

        answer = pipeline.process_query(query)
        print("\n🔹 Answer:", answer)

    print("\n👋 Exiting. Thank you!")

if __name__ == "__main__":
    main()
