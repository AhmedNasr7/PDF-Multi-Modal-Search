# 📄 AI-Powered Simple PDF Query System

An AI-powered system that allows users to **upload PDFs**, **query extracted content**, **extract and caption images** and **retrieve answers** using **vector search, ranking, and a locally-hosted vision-language model (Qwen2.5-VL)**.

## **🚀 Features**
✅ Upload PDFs & extract structured text + images  
✅ Store extracted text in a vector database (Qdrant)  
✅ Ask questions about the document & get accurate answers  
✅ Generate captions for images using Qwen2.5-VL  
✅ Interactive Streamlit UI + API for Vision-Language Model  

---

## **🛠️ Installation & Setup**

### **Option 1: Run with Docker Compose (Recommended)**
This method runs **both the Streamlit UI and the VLM API** in containers.

#### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/AhmedNasr7/PDF-Multi-Modal-Search
cd PDF-Multi-Modal-Search
```

#### **2️⃣ Build & Start Everything**
```bash
docker-compose up --build -d
```
- The `-d` flag runs it in **detached mode** (background).  
- This starts both **the Vision-Language Model API and the Streamlit UI**.

#### **3️⃣ Check Running Containers**
```bash
docker ps
```

#### **4️⃣ Stop & Remove Containers**
```bash
docker-compose down
```

#### **5️⃣ Access the Services**
- **📌 Streamlit UI:** [`http://localhost:8501`](http://localhost:8501)  
- **📌 API (VLM Model Server):** [`http://localhost:8000/caption_image`](http://localhost:8000/caption_image)  

---

### **Option 2: Run in a Conda Environment (Manual Setup)**
Use this method if you prefer to run everything manually.

#### **1️⃣ Create a Conda Environment**
```bash
conda create --name ai_pdf python=3.11 -y
conda activate ai_pdf
```

#### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **3️⃣ Run the VLM Server**
```bash
python server/vlm_server.py 
```
This starts the **Qwen2.5-VL Vision-Language Model API** at `http://localhost:8000/caption_image`.

#### **4️⃣ Run the Streamlit App**
```bash
streamlit run app.py --server.port=8501
```
The Streamlit UI will be available at `http://localhost:8501`.

Or Alternatively, use the CLI 
```bash
python main.py path/to/document.pdf 
```
---

## **📌 Usage Guide**

### **1️⃣ Upload & Process a PDF (Streamlit UI)**
1. Open the **Streamlit UI** (`http://localhost:8501`).
2. Upload a **PDF file**.
3. The system extracts **text & images** and stores embeddings.

### **2️⃣ Ask Questions About the Document (Streamlit UI)**
- Type your query in the input box.
- Click **"Search"** to get answers from the document.

---

### **Or: Process and Query PDFs via CLI (`main.py`)**
The **CLI version** allows processing and querying PDFs directly from the terminal.

#### **1️⃣ Process a PDF and Start Querying**
```bash
python main.py path/to/document.pdf --ranker tfidf --merger t5 --top_k 5
```
or
```bash
python main.py path/to/document.pdf --ranker cosine_similarity 
```
if no merger needed and no TF-IDF ranking needed.

- `path/to/document.pdf` → The PDF file to process.  
- `--ranker` → Ranking method (`tfidf`, `cosine_similarity`, `none`).  
- `--merger` → Answer merging method (`t5`, `concatenation`).  
- `--top_k` → Number of retrieved results (default: `5`).  

#### **2️⃣ Ask Questions via CLI**
Once the document is processed, you can start asking questions interactively:
```bash
📝 Your question: What are the applications of AI?
🔹 Answer: AI is used in various fields including healthcare, finance, and autonomous systems...
```

#### **3️⃣ Exit the CLI**
To exit the CLI session, simply type:
```bash
exit
```



---


