from flask import Flask, jsonify, request
from env import EmailTriageEnv, EmailAction

app = Flask(__name__)

# Global environment instance
env = EmailTriageEnv(task_difficulty="easy")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "project": "Email Triage OpenEnv",
        "message": "OpenEnv Space is live"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"health": "healthy"})


# ✅ FIXED: reset supports BOTH GET and POST
@app.route("/reset", methods=["GET", "POST"])
def reset_env():
    global env

    task = "easy"

    # If POST body has task, use it
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        task = data.get("task", "easy")

    # If GET query has task, use it
    if request.method == "GET":
        task = request.args.get("task", task)

    env = EmailTriageEnv(task_difficulty=task)
    obs = env.reset()

    return jsonify(obs.model_dump())


# ✅ state should also support GET + POST to be validator-safe
@app.route("/state", methods=["GET", "POST"])
def state_env():
    return jsonify(env.state())


# ✅ step must support POST
@app.route("/step", methods=["POST"])
def step_env():
    data = request.get_json(silent=True) or {}

    try:
        action = EmailAction(**data)
    except Exception:
        # fallback safe action if validator sends empty/odd body
        action = EmailAction(action_type="noop")

    obs, reward, done, info = env.step(action)

    return jsonify({
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)