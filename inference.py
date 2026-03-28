import json
from app.baseline import run_baseline

if __name__ == "__main__":
    result = run_baseline()
    print(json.dumps(result, indent=2))