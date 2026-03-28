<<<<<<< HEAD
import json
from app.baseline import run_baseline

if __name__ == "__main__":
    result = run_baseline()
=======
import json
from app.baseline import run_baseline

if __name__ == "__main__":
    result = run_baseline()
>>>>>>> 9016003d48f10a3716fa82c9a76038702b8d0558
    print(json.dumps(result, indent=2))