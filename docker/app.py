from flask import Flask, request, jsonify, send_from_directory
from rag import build_rag_chain, ask

app = Flask(__name__, static_folder="static", static_url_path="")

@app.before_first_request
def init():
    build_rag_chain(session_id="browser-session")


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

    if not user_input:
        return jsonify({"answer": "Please enter a message"}), 400

    answer = ask(user_input)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
