import subprocess
import os
import json

def start_skill_session(skill_name, user_data):
    workspace = f"./{skill_name}-workspace/iteration-1"
    os.makedirs(workspace, exist_ok=True)
    
    # 1. Capture Intent (Saving user context for the analyzer)
    with open(f"{workspace}/eval_metadata.json", "w") as f:
        json.dump(user_data, f)

    # 2. Run the Evaluator (Calling your existing script from your screenshot)
    # This generates the HTML viewer the user needs to see
    cmd = [
        "python", "eval-viewer/generate_review.py",
        workspace,
        "--skill-name", skill_name,
        "--static", f"{workspace}/review.html"
    ]
    
    subprocess.run(cmd)
    return f"{workspace}/review.html"

# Example: When user clicks 'Apply' on a gap, call this:
# path = start_skill_session("sustainability-policy", {"name": "User", "gap": "Policy Drafting"})