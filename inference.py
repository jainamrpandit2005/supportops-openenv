import os
import json
import textwrap
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

# Try to import from your local environment file
try:
    from env import EmailTriageEnv, EmailAction, EmailObservation
except ImportError:
    print("Error: env.py not found in the current directory.")

# Load environment variables from .env if it exists
load_dotenv()

# --- CONFIGURATION ---
# 1. PASTE YOUR KEY HERE:
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "PASTE_YOUR_KEY_HERE"

# 2. OTHER SETTINGS:
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
MAX_STEPS = 200
TEMPERATURE = 0.2
MAX_TOKENS = 150

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert email management assistant. Your task is to help triage and manage emails efficiently.
    
    For each email, you must decide on an action:
    - read: Read the current email
    - categorize: Categorize the email as 'work', 'personal', 'spam', or 'urgent'
    - mark_spam: Mark email as spam (for obvious spam emails)
    - mark_urgent: Mark important work emails as urgent
    - archive: Archive processed emails
    - noop: Do nothing for this step
    
    Respond with a JSON object containing:
    {
        "action_type": "read|categorize|mark_spam|mark_urgent|archive|noop",
        "email_id": "email_id_string (optional)",
        "category": "work|personal|spam|urgent (optional)",
        "folder": "folder_name (optional)"
    }
    
    Be strategic about categorization. Analyze email content carefully.
    """
).strip()


def build_user_prompt(observation: EmailObservation, history: List[str]) -> str:
    """Build the user prompt from observation and history"""
    email_info = ""
    if observation.current_email_content:
        email_info = f"\nCurrent Email:\n{observation.current_email_content}"
    
    inbox_status = f"\nInbox Status: {observation.processed_count}/{observation.email_count} processed"
    
    history_str = ""
    if history:
        history_str = "\nRecent Actions:\n" + "\n".join(history[-3:])
    
    prompt = textwrap.dedent(
        f"""
        {email_info}
        {inbox_status}
        {history_str}
        
        What action should you take next? Respond with a valid JSON object.
        """
    ).strip()
    
    return prompt


def parse_model_response(response_text: str) -> Optional[EmailAction]:
    """Parse the model's response into an EmailAction"""
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            action_dict = json.loads(json_str)
            return EmailAction(**action_dict)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse response: {e}")
        return EmailAction(action_type="noop")
    
    return EmailAction(action_type="noop")


def run_task(task_difficulty: str, client: OpenAI) -> Dict[str, Any]:
    """Run a single task and return results"""
    print(f"\n{'='*60}")
    print(f"Running Task: {task_difficulty.upper()}")
    print(f"{'='*60}")
    
    env = EmailTriageEnv(task_difficulty=task_difficulty)
    observation = env.reset()
    history: List[str] = []
    step_count = 0
    total_reward = 0.0
    
    while step_count < MAX_STEPS:
        step_count += 1
        user_prompt = build_user_prompt(observation, history)
        
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            
            response_text = completion.choices[0].message.content or ""
            action = parse_model_response(response_text)
            
        except Exception as e:
            print(f"Error calling model: {e}")
            action = EmailAction(action_type="noop")
        
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        action_log = f"Step {step_count}: {action.action_type}"
        if action.email_id: action_log += f" (email: {action.email_id})"
        if action.category: action_log += f" -> {action.category}"
        
        print(action_log)
        print(f"  Reward: {reward:+.2f} | Total: {total_reward:+.2f}")
        
        history.append(action_log)
        if done:
            print(f"Task completed in {step_count} steps")
            break
    
    final_score = info.get("final_score", 0.0)
    return {
        "difficulty": task_difficulty,
        "score": final_score,
        "reward": total_reward,
        "steps": step_count,
    }


def main():
    """Main inference script"""
    print("Email Triage Environment - Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    
    # Error Handling: Check if the key is actually set
    if not API_KEY or API_KEY == "PASTE_YOUR_KEY_HERE":
        print("\n!!! ERROR: API KEY MISSING !!!")
        print("Please edit line 17 and paste your OpenAI API key.")
        return

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    results = []
    for difficulty in ["easy", "medium", "hard"]:
        try:
            result = run_task(difficulty, client)
            results.append(result)
        except Exception as e:
            print(f"Error running {difficulty} task: {e}")
            results.append({
                "difficulty": difficulty,
                "score": 0.0,
                "reward": 0.0,
                "steps": 0,
                "error": str(e),
            })
    
    print(f"\n{'='*60}\nFINAL RESULTS SUMMARY\n{'='*60}")
    for result in results:
        print(f"\n{result['difficulty'].upper()}:")
        print(f"  Score: {result['score']:.2f} | Reward: {result['reward']:+.2f} | Steps: {result['steps']}")
    
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\nAverage Score: {avg_score:.2f}")

if __name__ == "__main__":
    main()