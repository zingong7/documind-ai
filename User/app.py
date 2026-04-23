import boto3 
import streamlit as st
import os
import uuid
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader    
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_aws import BedrockLLM
from langchain.chains import RetrievalQA

s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

folder_path = "/tmp/"

def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="myfaiss.faiss", Filename=f"{folder_path}myfaiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="myfaiss.pkl", Filename=f"{folder_path}myfaiss.pkl")

def get_llm():
    llm = BedrockLLM(
        model_id="us.meta.llama3-1-8b-instruct-v1:0",
        client=bedrock_client,
        model_kwargs={'max_gen_len': 512}
    )
    return llm

def get_response(llm, vectorstore, question):
    prompt_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant. Use the given context to answer the question concisely.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Context: {context}

    Question: {question}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": question})
    return answer['result']

def main():
    st.set_page_config(
        page_title="DocuMind AI — Chat",
        page_icon="💬",
        layout="centered"
    )

    # Header
    st.title("DocuMind AI")
    st.subheader("Chat with your PDF")
    st.caption("Ask any question about your uploaded document. Powered by AWS Bedrock and Llama 3.")
    st.divider()

    # Load index
    with st.spinner("Loading knowledge base from S3..."):
        try:
            load_index()
            faiss_index = FAISS.load_local(
                index_name='myfaiss',
                folder_path=folder_path,
                embeddings=bedrock_embedding,
                allow_dangerous_deserialization=True
            )
            st.success("Knowledge base loaded and ready.")
        except Exception as e:
            st.error(f"Failed to load knowledge base: {str(e)}")
            st.warning("Make sure you have uploaded and processed a PDF from the Admin panel first.")
            st.stop()

    st.divider()

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation")
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
        st.divider()

    # Question input
    st.subheader("Ask a Question")
    question = st.text_input("Type your question here", placeholder="e.g. What is the main topic of this document?")

    col1, col2 = st.columns([3, 1])
    with col1:
        ask = st.button("Ask Question", type="primary", use_container_width=True)
    with col2:
        clear = st.button("Clear Chat", use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if ask:
        if not question.strip():
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Searching knowledge base and generating answer..."):
                try:
                    llm = get_llm()
                    answer = get_response(llm, faiss_index, question)

                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer
                    })

                    # Show latest answer
                    st.divider()
                    st.subheader("Answer")
                    with st.chat_message("assistant"):
                        st.write(answer)
                    st.success("Done.")

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

    st.divider()
    st.caption("DocuMind AI — Powered by AWS Bedrock · LangChain · FAISS · Llama 3")

if __name__ == "__main__":
    main()