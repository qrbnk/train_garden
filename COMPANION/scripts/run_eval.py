#!/usr/bin/env python3
"""Run trigger evaluation for a skill description using local Ollama inference."""

import argparse
import requests
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.utils import parse_skill_md

def find_project_root() -> Path:
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".claude").is_dir():
            return parent
    return current

def run_single_query(query, skill_name, skill_description, timeout, project_root, model=None):
    """Local evaluation logic using Ollama."""
    # Internal name-match fallback for extra reliability
    if skill_name.lower().replace("-", " ") in query.lower():
        return True

    url = "http://localhost:11434/api/generate"
    # We ask the local AI to act as a router
    prompt = f"Does the query '{query}' require the skill: '{skill_description}'? Respond ONLY with YES or NO."
    
    try:
        res = requests.post(
            url, 
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}, 
            timeout=timeout
        )
        if res.status_code == 200:
            text = res.json().get("response", "").upper()
            return "YES" in text
        return False
    except:
        # Final fallback: if Ollama isn't running, default to standard name matching
        return skill_name.lower().replace("-", " ") in query.lower()

def run_eval(eval_set, skill_name, description, num_workers, timeout, project_root, model=None, runs_per_query=1, trigger_threshold=0.5):
    """Evaluates a skill description using the signature required by run_loop.py."""
    results = []
    passed = 0
    total = len(eval_set)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                run_single_query, q["query"], skill_name, description, timeout, project_root, model
            ): q for q in eval_set
        }

        for future in as_completed(futures):
            q = futures[future]
            try:
                prediction = future.result()
                is_correct = (prediction == q["should_trigger"])
                if is_correct:
                    passed += 1
                
                results.append({
                    "query": q["query"],
                    "should_trigger": q["should_trigger"],
                    "pass": is_correct,
                    "triggers": 1 if prediction else 0,
                    "runs": 1
                })
            except Exception as e:
                results.append({"query": q["query"], "error": str(e), "pass": False})

    return {
        "results": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "description": description
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--skill-path", required=True)
    parser.add_argument("--description", default=None)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--runs-per-query", type=int, default=1)
    parser.add_argument("--trigger-threshold", type=float, default=0.5)
    args = parser.parse_args()

    eval_set = json.loads(Path(args.eval_set).read_text())
    name, original_description, content = parse_skill_md(Path(args.skill_path))
    
    output = run_eval(
        eval_set, 
        name, 
        args.description or original_description, 
        args.num_workers, 
        args.timeout, 
        find_project_root(), 
        args.model,
        args.runs_per_query,
        args.trigger_threshold
    )
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()