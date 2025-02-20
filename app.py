import os
import streamlit as st
from backend import load_pdfs, create_faiss_index, get_response

# Ensure the 'uploads' directory exists
os.makedirs("uploads", exist_ok=True)

# Streamlit UI
st.set_page_config(page_title="Chat with PDFs", layout="wide")
st.title("Chat with your PDFs 📄🤖")

# File upload
uploaded_files = st.file_uploader(
    "Upload 5-10 PDFs", type=["pdf"], accept_multiple_files=True
)

# Process uploaded PDFs
if st.button("Process PDFs"):
    if uploaded_files:
        pdf_paths = []

        # Show a loading spinner
        with st.spinner("Processing PDFs..."):
            for uploaded_file in uploaded_files:
                file_path = os.path.join("uploads", uploaded_file.name)

                # Check if file already exists to avoid redundant processing
                if not os.path.exists(file_path):
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                pdf_paths.append(file_path)

            # Process PDFs and create FAISS index
            try:
                chunks = load_pdfs(pdf_paths)
                create_faiss_index(chunks)
                st.success("✅ PDFs processed successfully! You can now ask questions.")
            except Exception as e:
                st.error(f"❌ Error processing PDFs: {e}")
    else:
        st.warning("⚠️ Please upload at least one PDF before processing.")

# Chat input
query = st.text_input("Ask a question:")
if query:
    with st.spinner("Generating answer..."):
        try:
            response = get_response(query)
            st.write("### 🤖 Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Error: {e}")
