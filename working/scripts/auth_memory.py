import hashlib
import json
import os
from pathlib import Path

def get_user_session(email):
    """
    Simulates a Magic Link registration. 
    Returns a unique user_id and path to their persistent memory.
    """
    user_id = hashlib.sha256(email.lower().strip().encode()).hexdigest()[:12]
    user_dir = Path(f"COMPANION/users/{user_id}")
    user_dir.mkdir(parents=True, exist_ok=True)
    
    memory_file = user_dir / "memory.json"
    if not memory_file.exists():
        initial_memory = {
            "user_email": email,
            "preferences": {
                "depth": "technical",
                "focus_areas": [], # e.g., ['volunteering', 'courses']
                "learning_style": "interactive"
            },
            "history": []
        }
        memory_file.write_text(json.dumps(initial_memory, indent=2))
    
    return user_id, memory_file

def update_user_intent(memory_file, new_intent):
    """Call this when the user says 'I want to find courses'"""
    memory = json.loads(memory_file.read_text())
    if new_intent not in memory["preferences"]["focus_areas"]:
        memory["preferences"]["focus_areas"].append(new_intent)
    memory_file.write_text(json.dumps(memory, indent=2))