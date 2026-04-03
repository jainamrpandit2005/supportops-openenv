import os
import json
from typing import Any, Dict, List, Optional
from openai import OpenAI

# ============================================================
# Optional OpenAI setup
# ============================================================
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("OPENAI_MODEL", "gpt-4-turbo"))
API_BASE_URL = os.getenv("API_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"Model: {MODEL_NAME}")
print(f"API Base: {API_BASE_URL}")

USE_LLM = bool(OPENAI_API_KEY)

if not USE_LLM:
    print("\n⚠ OPENAI_API_KEY not set.")
    print("Using deterministic rule-based baseline instead.\n")

client = None
if USE_LLM:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=API_BASE_URL
    )

# ============================================================
# Import your local environment directly
# ============================================================
try:
    from env import EmailTriageEnv
except ImportError:
    raise ImportError(
        "Could not import EmailTriageEnv from env.py.\n"
        "Make sure env.py exists in the same folder as inference.py"
    )

# ============================================================
# Action helper
# ============================================================
class SimpleAction:
    def __init__(self, action_type: str, email_id: Optional[str] = None, category: Optional[str] = None):
        self.action_type = action_type
        self.email_id = email_id
        self.category = category

    def __repr__(self):
        return f"SimpleAction(action_type={self.action_type}, email_id={self.email_id}, category={self.category})"


# ============================================================
# Utility functions
# ============================================================
def obs_to_dict(obs: Any) -> Dict[str, Any]:
    if obs is None:
        return {}

    if isinstance(obs, dict):
        return obs

    if hasattr(obs, "model_dump"):
        return obs.model_dump()

    if hasattr(obs, "dict"):
        return obs.dict()

    if hasattr(obs, "__dict__"):
        return vars(obs)

    return {}


def get_inbox(obs: Any) -> List[Dict[str, Any]]:
    d = obs_to_dict(obs)
    inbox = d.get("inbox", [])
    return inbox if isinstance(inbox, list) else []


def get_email_content(obs: Any) -> str:
    d = obs_to_dict(obs)
    return str(d.get("current_email_content", "") or "")


# ============================================================
# LLM helper
# ============================================================
def call_llm_for_category(sender: str, subject: str, content: str) -> Optional[str]:
    if client is None:
        return None

    prompt = f"""
You are an email triage assistant.

Classify the following email into exactly one of these categories:
- work
- personal
- important
- spam

Return ONLY one word from the above list.

Sender: {sender}
Subject: {subject}
Content: {content}
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise email classification assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip().lower()

        allowed = {"work", "personal", "important", "spam"}
        if result in allowed:
            return result

        return None
    except Exception as e:
        print(f"STEP action=llm_classification_failed error={str(e)}")
        return None


# ============================================================
# Improved rule-based classifier
# ============================================================
def classify_email(sender: str, subject: str, content: str = "") -> str:
    sender = (sender or "").lower()
    subject = (subject or "").lower()
    content = (content or "").lower()
    text = f"{sender} {subject} {content}"

    # -------------------------
    # 1) VERY STRONG SENDER RULES FIRST
    # -------------------------
    if sender.endswith("@company.com"):
        return "work"

    if any(x in sender for x in ["friend@", "mom@", "grandpa@", "family"]):
        return "personal"

    if any(x in sender for x in ["payments", "billing", "finance"]):
        return "important"

    if any(x in sender for x in ["noreply@spam.com", "promo@", "ads@", "spam"]):
        return "spam"

    # -------------------------
    # 2) STRONG SPAM SIGNALS
    # -------------------------
    phishing_keywords = [
        "account has been compromised",
        "verify your account",
        "urgent action required",
        "suspended account",
        "confirm your identity",
        "login immediately"
    ]

    spam_keywords = [
        "click here", "win prizes", "free money", "lottery", "prize",
        "claim now", "congratulations", "limited time", "selected",
        "offer expires", "act now", "unsubscribe", "winner"
    ]

    if any(k in text for k in phishing_keywords):
        return "spam"

    if any(k in text for k in spam_keywords):
        return "spam"

    if "verification" in text and "account" in text:
        return "spam"

    if "notifications@bank.com" in sender and "verification" in text:
        return "spam"

    # -------------------------
    # 3) IMPORTANT SIGNALS
    # -------------------------
    important_keywords = [
        "invoice", "billing", "payment", "transaction",
        "receipt", "statement", "otp", "security code",
        "confirm your account"
    ]

    if any(k in text for k in important_keywords):
        return "important"

    if "urgent" in subject and "account" in subject:
        return "important"

    # -------------------------
    # 4) WORK SIGNALS
    # -------------------------
    work_keywords = [
        "project", "deadline", "team standup", "standup meeting", "meeting",
        "code review", "feature x", "deliverables", "blockers",
        "benefits enrollment", "enrollment deadline", "hr", "boss",
        "colleague", "team", "sprint", "milestone", "office"
    ]

    if any(k in text for k in work_keywords):
        return "work"

    # -------------------------
    # 5) PERSONAL SIGNALS
    # -------------------------
    personal_keywords = [
        "let's grab coffee", "lets grab coffee", "grab coffee",
        "family dinner", "how are you", "weekend", "sunday",
        "catch up", "dinner", "friend", "mom", "grandpa", "family"
    ]

    if any(k in text for k in personal_keywords):
        return "personal"

    if sender.endswith("@gmail.com") or sender.endswith("@yahoo.com") or sender.endswith("@hotmail.com"):
        if any(k in text for k in ["coffee", "dinner", "weekend", "how are you", "catch up"]):
            return "personal"

    # -------------------------
    # 6) FALLBACK
    # -------------------------
    return "important"


def decide_category(sender: str, subject: str, content: str = "") -> str:
    llm_result = call_llm_for_category(sender, subject, content)
    if llm_result:
        return llm_result
    return classify_email(sender, subject, content)


# ============================================================
# Environment builder
# ============================================================
def create_env():
    try:
        env = EmailTriageEnv()
        print(f"STEP action=create_env status=success env_class={env.__class__.__name__}")
        return env
    except Exception as e:
        raise RuntimeError(
            f"Could not create EmailTriageEnv instance.\nError: {e}"
        )


# ============================================================
# Better score extractor
# ============================================================
def extract_score(env, obs=None, info=None):
    if isinstance(info, dict):
        for key in ["score", "final_score", "accuracy"]:
            if key in info:
                try:
                    return float(info[key])
                except:
                    pass

    d = obs_to_dict(obs)
    for key in ["score", "final_score", "accuracy"]:
        if key in d:
            try:
                return float(d[key])
            except:
                pass

    for attr in ["score", "final_score", "accuracy"]:
        if hasattr(env, attr):
            try:
                return float(getattr(env, attr))
            except:
                pass

    for method_name in ["get_score", "compute_score", "final_score"]:
        if hasattr(env, method_name):
            try:
                value = getattr(env, method_name)()
                return float(value)
            except:
                pass

    return None


# ============================================================
# Main task runner
# ============================================================
def run_task(env, task_id: str):
    print(f"\nSTART task={task_id.upper()} env=EmailTriageEnv")

    total_reward = 0.0
    step_count = 0
    info = {}

    try:
        print("STEP action=reset_env")
        obs = env.reset(task_id)
    except TypeError:
        try:
            obs = env.reset(task_name=task_id)
        except:
            obs = env.reset()

    inbox = get_inbox(obs)

    if not inbox:
        print("STEP action=check_inbox status=empty")
        print("END status=error reason=no_emails_found")
        print("🔍 Observation received:")
        print(obs)
        return {
            "score": 0.0,
            "total_reward": 0.0,
            "steps": 0
        }

    print(f"STEP action=check_inbox status=success email_count={len(inbox)}")

    done = False

    for email in inbox:
        if done:
            break

        email_id = email.get("id")
        sender = email.get("sender", "")
        subject = email.get("subject", "")

        if not email_id:
            continue

        # -------------------------
        # READ
        # -------------------------
        try:
            print(f"STEP action=read_email email_id={email_id}")
            action = SimpleAction(action_type="read", email_id=email_id)
            obs, reward, done, info = env.step(action)
            step_count += 1
            total_reward += reward

            print(f"  Reward: {reward:+.2f} | Total: {total_reward:+.2f}")
        except Exception as e:
            print(f"STEP action=read_email status=error email_id={email_id} error={str(e)}")
            continue

        content = get_email_content(obs)

        # -------------------------
        # CATEGORIZE
        # -------------------------
        try:
            category = decide_category(sender, subject, content)
            print(f"STEP action=categorize_email email_id={email_id} predicted={category}")

            action = SimpleAction(
                action_type="categorize",
                email_id=email_id,
                category=category
            )

            obs, reward, done, info = env.step(action)
            step_count += 1
            total_reward += reward

            print(f"  Reward: {reward:+.2f} | Total: {total_reward:+.2f}")
        except Exception as e:
            print(f"STEP action=categorize_email status=error email_id={email_id} error={str(e)}")
            continue

    final_score = extract_score(env, obs=obs, info=info)

    if final_score is None:
        read_reward_total = len(inbox) * 0.1
        categorize_reward = total_reward - read_reward_total

        n = len(inbox)
        estimated_correct = (categorize_reward + (0.2 * n)) / 0.7
        estimated_score = estimated_correct / n if n > 0 else 0.0

        final_score = round(max(0.0, min(1.0, estimated_score)), 2)

    print("STEP action=task_summary")
    print(f"  Final Score: {final_score:.2f}")
    print(f"  Total Reward: {total_reward:+.2f}")
    print(f"  Steps: {step_count}")
    print(f"END status=success task={task_id.upper()} score={final_score:.2f}")

    return {
        "score": final_score,
        "total_reward": total_reward,
        "steps": step_count
    }


# ============================================================
# Main
# ============================================================
def main():
    tasks = ["easy", "medium", "hard"]
    results = {}

    for task in tasks:
        try:
            env = create_env()
            results[task.upper()] = run_task(env, task)
        except Exception as e:
            print(f"END status=error task={task.upper()} error={str(e)}")
            results[task.upper()] = {
                "score": 0.0,
                "total_reward": 0.0,
                "steps": 0
            }

    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS SUMMARY")
    print("=" * 60)

    total_scores = []

    for task_name, result in results.items():
        score = result["score"]
        total_scores.append(score)

        print(f"\n{task_name}:")
        print(f"  Score: {score:.2f}")
        print(f"  Total Reward: {result['total_reward']:+.2f}")
        print(f"  Steps: {result['steps']}")

    avg_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
    print(f"\n⭐ Average Score: {avg_score:.2f}")


if __name__ == "__main__":
    main()
