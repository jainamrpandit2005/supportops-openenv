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


@app.route("/reset", methods=["GET", "POST"])
def reset_env():
    global env

    task = "easy"

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        task = data.get("task", "easy")

    if request.method == "GET":
        task = request.args.get("task", task)

    env = EmailTriageEnv(task_difficulty=task)
    obs = env.reset()

    return jsonify(obs.model_dump())


@app.route("/state", methods=["GET", "POST"])
def state_env():
    return jsonify(env.state())


@app.route("/step", methods=["POST"])
def step_env():
    data = request.get_json(silent=True) or {}

    try:
        action = EmailAction(**data)
    except Exception:
        action = EmailAction(action_type="noop")

    obs, reward, done, info = env.step(action)

    return jsonify({
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    })


def main():
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()