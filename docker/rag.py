import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_message_histories import RedisChatMessageHistory

load_dotenv("/app/.env")

rag_chain = None
chat_history = None


def build_rag_chain(session_id="web-session"):
    global rag_chain, chat_history

    source_url = os.getenv("SOURCE_URL")
    redis_url = os.getenv("REDIS_URL")

    chat_history = RedisChatMessageHistory(
        session_id=session_id,
        url=redis_url,
        ttl=3600
    )

    docs = WebBaseLoader(source_url).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="/data/chroma"
    )

    retriever = vectordb.as_retriever()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use CONTEXT and HISTORY. "
         "If unrelated, answer normally."),
        ("human",
         "HISTORY:\n{history}\n\n"
         "QUESTION: {question}\n\n"
         "CONTEXT:\n{context}")
    ])

    rag_chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": RunnablePassthrough(),
            "history": lambda _: "\n".join(
                f"{m.type.upper()}: {m.content}"
                for m in chat_history.messages
            )
        }
        | prompt
        | llm
    )


def ask(question: str) -> str:
    result = rag_chain.invoke({"question": question})

    chat_history.add_user_message(question)
    chat_history.add_ai_message(result.content)

    return result.content
