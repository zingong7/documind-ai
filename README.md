# DocuMind AI — Chat with PDF using AWS Bedrock

A RAG (Retrieval Augmented Generation) application that lets you upload PDF documents and chat with them using AWS Bedrock and Llama 3.

---

## Architecture

- **Admin App** — Upload PDFs, generate vector embeddings and store FAISS index in S3
- **Client App** — Load index from S3 and chat with your document using natural language

---

## Tech Stack

- Python 3.11
- Streamlit
- AWS Bedrock (Amazon Titan Embeddings + Llama 3)
- LangChain
- FAISS
- Docker
- Amazon S3

---

## Project Structure

```
├── admin.py              # Admin panel for uploading PDFs
├── app.py                # Client chat interface
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker config for admin
├── Dockerfile.client     # Docker config for client
├── .env.example          # Example environment variables
└── .streamlit/
    └── config.toml       # Streamlit theme config
```

---

## Prerequisites

- AWS Account with Bedrock access enabled
- S3 bucket created
- Docker installed
- IAM user with the following permissions:
  - `AmazonBedrockFullAccess`
  - `AmazonS3FullAccess`

---

## Environment Variables

Create a `.env` file in the root of the project:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
BUCKET_NAME=your_bucket_name
```

---

## Run with Docker

### Admin App

```bash
docker build -t pdf-reader-admin .
docker run --env-file .env -p 8083:8083 -it pdf-reader-admin
```

Open at: http://localhost:8083

### Client App

```bash
docker build -t pdf-reader-client -f Dockerfile.client .
docker run --env-file .env -p 8084:8084 -it pdf-reader-client
```

Open at: http://localhost:8084

---

## Usage

1. Open the **Admin App** at http://localhost:8083
2. Upload a PDF document
3. Click **Build Knowledge Base** and wait for it to complete
4. Open the **Client App** at http://localhost:8084
5. Ask any question about your document

---

## AWS Bedrock Models Used

| Model | Purpose |
|-------|---------|
| `amazon.titan-embed-text-v2:0` | Generate vector embeddings |
| `us.meta.llama3-1-8b-instruct-v1:0` | Generate answers |

---

## How It Works

1. PDF is uploaded and parsed into pages
2. Pages are split into chunks using LangChain's RecursiveCharacterTextSplitter
3. Each chunk is embedded using Amazon Titan Embeddings via AWS Bedrock
4. Embeddings are stored in a FAISS vector index and uploaded to S3
5. On the client side, the index is downloaded from S3
6. User questions are matched against the index using similarity search
7. Relevant chunks are passed to Llama 3 on AWS Bedrock to generate an answer

---

## Requirements

```
streamlit
pypdf
langchain
langchain-aws
langchain-community
langchain-core
langchain-text-splitters
faiss-cpu
boto3
```
