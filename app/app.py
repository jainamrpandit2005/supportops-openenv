from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "project": "Email Triage OpenEnv",
        "message": "Hugging Face Space is live"
    })

@app.route("/reset")
def reset():
    return jsonify({
        "status": "ok",
        "message": "reset endpoint working"
    })

@app.route("/health")
def health():
    return jsonify({
        "health": "healthy"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)