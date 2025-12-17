import os
from dotenv import load_dotenv
load_dotenv() 

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import RedisChatMessageHistory


def get_chat_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.getenv("REDIS_URL")
    )


def build_dynamic_rag(url: str):
    docs = WebBaseLoader(url).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer ONLY using the provided context from the webpage."),
        ("human",
         "QUESTION:\n{question}\n\nCONTEXT:\n{context}")
    ])

    return (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
