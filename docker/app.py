import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from utils import extract_url
from langchain_groq import ChatGroq
from rag import build_dynamic_rag, get_chat_history

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    session_id = data.get("session_id", "browser-session")

    chat_history = get_chat_history(session_id)
    chat_history.add_user_message(user_input)

    url = extract_url(user_input)

    if url:
        # 🔹 URL-based RAG
        rag = build_dynamic_rag(url)
        answer = rag.invoke(user_input).content
    else:
        # 🔹 Normal chat (NO RAG)
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0
        )
        answer = llm.invoke(user_input).content

    chat_history.add_ai_message(answer)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
