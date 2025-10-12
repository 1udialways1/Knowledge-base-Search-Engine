import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file using PyMuPDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")  # explicit mode
    return text.strip()


def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """Splits text into overlapping chunks for embedding."""
    if not text.strip():
        raise ValueError("Input text is empty. Cannot create chunks.")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)


def create_vector_store(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Creates a FAISS vector store from text chunks."""
    if not chunks:
        raise ValueError("No text chunks provided for vector store creation.")
    
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store
