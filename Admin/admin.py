import boto3 
import streamlit as st
import os
import uuid
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader    
from langchain_community.vectorstores import FAISS

s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embedding)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)
    s3_client.upload_file(Filename=folder_path + file_name + ".faiss", Bucket=BUCKET_NAME, Key="myfaiss.faiss")
    s3_client.upload_file(Filename=folder_path + file_name + ".pkl", Bucket=BUCKET_NAME, Key="myfaiss.pkl")
    return True

def main():
    st.set_page_config(
        page_title="DocuMind AI — Admin",
        page_icon="📄",
        layout="centered"
    )

    # Header
    st.title("DocuMind AI")
    st.subheader("Admin Panel — Knowledge Base Builder")
    st.caption("Upload a PDF to generate a vector index powered by AWS Bedrock and store it in S3.")
    st.divider()

    # Pipeline overview
    st.subheader("Pipeline")
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**Step 1**\n\nUpload PDF")
    col2.info("**Step 2**\n\nParse & Split")
    col3.info("**Step 3**\n\nEmbed via Titan")
    col4.info("**Step 4**\n\nSave to S3")
    st.divider()

    # Upload
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.divider()

        # File details
        st.subheader("File Details")
        file_size = len(uploaded_file.getvalue()) / 1024
        col1, col2, col3 = st.columns(3)
        col1.metric("File Name", uploaded_file.name if len(uploaded_file.name) < 18 else uploaded_file.name[:15] + "...")
        col2.metric("Size", f"{file_size:.1f} KB")
        col3.metric("Type", "PDF")

        # Save file
        request_id = get_unique_id()
        st.caption(f"Request ID: `{request_id}`")

        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode='wb') as w:
            w.write(uploaded_file.getvalue())

        # Parse
        with st.spinner("Reading and parsing PDF..."):
            loader = PyPDFLoader(saved_file_name)
            pages = loader.load_and_split()

        # Split
        with st.spinner("Splitting text into chunks..."):
            split_docs = split_text(pages, 1000, 200)

        avg_chunk = round(sum(len(d.page_content) for d in split_docs) / len(split_docs)) if split_docs else 0

        st.divider()

        # Document stats
        st.subheader("Document Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pages", len(pages))
        col2.metric("Text Chunks", len(split_docs))
        col3.metric("Avg Chunk", f"{avg_chunk} chars")

        st.divider()

        # Build
        st.subheader("Build Knowledge Base")
        st.caption("Generates embeddings using Amazon Titan and uploads the FAISS index to S3.")

        if st.button("Build Knowledge Base", type="primary", use_container_width=True):
            progress = st.progress(0, text="Initializing...")
            progress.progress(25, text="Generating vector embeddings...")
            result = create_vector_store(request_id, split_docs)
            progress.progress(80, text="Uploading index to S3...")
            progress.progress(100, text="Complete.")

            if result:
                st.success("Knowledge base built and uploaded to S3 successfully.")
                st.info("Open the client app to start chatting with your document.")
            else:
                st.error("Something went wrong. Please check the logs.")
    else:
        st.warning("No file uploaded yet. Choose a PDF above to get started.")

    st.divider()
    st.caption("DocuMind AI — Powered by AWS Bedrock · LangChain · FAISS")

if __name__ == "__main__":
    main()