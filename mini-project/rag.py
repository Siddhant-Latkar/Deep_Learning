import os
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "DATA")

logging.basicConfig(level=logging.INFO)

def load_pdfs():
    docs = []

    logging.info(f"PDF_FOLDER: {PDF_FOLDER}")

    if not os.path.exists(PDF_FOLDER):
        raise FileNotFoundError(f"Folder not found: {PDF_FOLDER}")

    files = os.listdir(PDF_FOLDER)
    logging.info(f"Files found: {files}")

    for file in files:
        if file.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, file)
            loader = PyPDFLoader(path)
            pages = loader.load()

            for p in pages:
                p.metadata["source"] = file

            docs.extend(pages)

    return docs

def create_rag():
    documents = load_pdfs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful student assistant.

Answer clearly using the given context.
If not found, say: "Not in documents".

Context:
{context}

Question:
{question}
""")

    return retriever, llm, prompt
    
def ask_question(query, retriever, llm, prompt):
    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    response = llm.invoke(
        prompt.format(context=context, question=query)
    )

    return response.content
