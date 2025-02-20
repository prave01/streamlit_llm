import os
import google.generativeai as genai
import faiss
import pickle
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import google.auth
from google.auth.credentials import AnonymousCredentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("AIzaSyBSPAjN09kQE0QrBMFjQX5mMwbTflWzU44")

# Initialize Gemini AI
genai.configure(api_key="AIzaSyBSPAjN09kQE0QrBMFjQX5mMwbTflWzU44")  # Replace with your actual API key

# Function to load and split PDF text
def load_pdfs(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    return chunks

# Function to create embeddings and store in FAISS
def create_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save the FAISS index
    faiss.write_index(vectorstore.index, "vectorstore.index")
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

# Function to load FAISS index
def load_faiss_index():
    if os.path.exists("vectorstore.index"):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index = faiss.read_index("vectorstore.index")
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
            vectorstore.index = index
        return vectorstore
    return None

# Function to retrieve context from FAISS
def retrieve_context(query):
    vectorstore = load_faiss_index()
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        return context
    return "No relevant information found."

# Function to query Google Gemini AI
def query_gemini(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Function to generate response
def get_response(query):
    context = retrieve_context(query)
    prompt = f"Answer based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    return query_gemini(prompt)
