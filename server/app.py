import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from env import EmailTriageEnv, EmailAction

app = Flask(__name__)

# Global environment instance
env_instance = EmailTriageEnv(task_difficulty="easy")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Email Triage OpenEnv is running!",
        "status": "ok"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/reset", methods=["GET", "POST"])
def reset_env():
    global env_instance

    difficulty = "easy"

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        difficulty = data.get("task_id", data.get("difficulty", "easy"))
    else:
        difficulty = request.args.get("task_id", "easy")

    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "easy"

    env_instance = EmailTriageEnv(task_difficulty=difficulty)
    obs = env_instance.reset()

    return jsonify(obs.model_dump())


@app.route("/step", methods=["POST"])
def step_env():
    global env_instance
    data = request.get_json(silent=True) or {}

    try:
        action = EmailAction(**data)
    except Exception as e:
        return jsonify({"error": f"Invalid action format: {str(e)}"}), 400

    obs, reward, done, info = env_instance.step(action)

    return jsonify({
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    })


@app.route("/state", methods=["GET"])
def get_state():
    global env_instance
    return jsonify(env_instance.state())


def main():
    """Start the OpenEnv server."""
    app.run(host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
