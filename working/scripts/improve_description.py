#!/usr/bin/env python3
import argparse
import requests
import json
import re
from pathlib import Path
from scripts.utils import parse_skill_md

def _call_ai(prompt, model, timeout=60):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        return response.json().get("response", "").strip() if response.status_code == 200 else ""
    except Exception as e:
        print(f"⚠️ Local AI Error: {e}")
        return ""

def improve_description(skill_name, skill_content, current_description, eval_results, history, model):
    # COMPREHENSIVE FEEDBACK: Aggregate all failures from the current run
    failures = []
    if history:
        for entry in history:
            # Safely get results from previous iterations
            results = entry.get('results', [])
            failures.extend([r['query'] for r in results if not r.get('pass', False)])

    # If this is the first failure, use the current results
    if not failures:
        failures = [r['query'] for r in eval_results.get('results', []) if not r.get('pass', False)]

    prompt = f"""
    [INST] <<SYS>> You are a Technical Optimizer. <</SYS>>
    SKILL: {skill_name}
    CODE CONTEXT: {skill_content[:1000]}
    
    FAILURES TO FIX: {list(set(failures))}
    
    TASK: Your previous description failed the queries above. 
    Write a 1-sentence description starting with "Use this skill to..." that is more technically precise.
    [/INST]
    """
    # ... (rest of the call_ai and cleaning logic) ...    # COMPREHENSIVE FEEDBACK: Collect every query that has EVER failed in this session
    all_past_failures = []
    for entry in history:
        failures = [r['query'] for r in entry.get('results', []) if not r.get('pass')]
        all_past_failures.extend(failures)

    prompt = f"""
    [INST] <<SYS>> You are a Technical Optimizer. Output ONLY the description. <</SYS>>
    SKILL: {skill_name}
    FULL KNOWLEDGE BASE: {skill_content[:1500]}
    
    USER FEEDBACK (PREVIOUS ERRORS TO FIX):
    {list(set(all_past_failures))}
    
    TASK: Your previous description "{current_description}" was inaccurate.
    Write a new 1-sentence description starting with "Use this skill to...".
    Make it technical and specific so it avoids the errors listed above.
    [/INST]
    """
    
    ai_text = _call_ai(prompt, model)
    # Cleaning the response from any AI chatter
    clean_desc = ai_text.split("[/INST]")[-1].strip()
    clean_desc = re.sub(r'^(Description|NEW DESCRIPTION|SKILL NAME):', '', clean_desc, flags=re.IGNORECASE).strip()
    return clean_desc

def generate_description_from_content(skill_name, skill_content):
    """Generates a high-quality description based on the actual skill content."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--skill-path", required=True)
    parser.add_argument("--history", default="[]")
    args = parser.parse_args()

    eval_results = json.loads(Path(args.eval_results).read_text())
    history = json.loads(args.history) if args.history else []
    name, _, content = parse_skill_md(Path(args.skill_path))
    
    new_desc = improve_description(name, content, "", eval_results, history, "llama3.2:1b")
    print(json.dumps({"description": new_desc}))

if __name__ == "__main__":
    main()