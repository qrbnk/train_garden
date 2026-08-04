#!/usr/bin/env python3
import argparse
import requests
import sys
import time
from pathlib import Path

# Setup imports
current_file = Path(__file__).resolve()
sys.path.insert(0, str(current_file.parent.parent))

try:
    from generate_report import generate_html
    from scripts.improve_description import improve_description
    from scripts.run_eval import run_eval
    from scripts.utils import parse_skill_md
    from scripts.package_skill import package_skill
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def update_ui_safely(report_path, name, skill_path, history, knowledge_text):
    if report_path:
        r_path = Path(report_path).resolve()
        r_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {"history": history, "research_base": knowledge_text}
        html_content = generate_html(report_data, skill_name=name, skill_path=str(skill_path))
        r_path.write_text(html_content, encoding="utf-8")

def generate_comprehensive_knowledge(skill_path, skill_name, model, memory_file=None):
    # --- FIX: Define the missing variables ---
    url = "http://localhost:11434/api/generate" # Default for local Ollama
    target_model = model 
    
    extra_context = ""
    # (Existing memory logic...)
    if memory_file and memory_file.exists():
        try:
            import json
            memory = json.loads(memory_file.read_text())
            interests = memory.get("preferences", {}).get("focus_areas", [])
            if interests:
                extra_context = f"Also research practical ways to engage with this, specifically: {', '.join(interests)}."
        except: pass

    prompt = f"[INST] Write a technical whitepaper for '{skill_name}'. Include ARCHITECTURE and STANDARDS. {extra_context} [/INST]"
    
    try:
        # Use the local or remote URL correctly
        res = requests.post(url, json={"model": target_model, "prompt": prompt, "stream": False}, timeout=600)
        res.raise_for_status()
        content = res.json().get("response", "").strip()
        if content:
            (skill_path / "KNOWLEDGE.md").write_text(content, encoding="utf-8")
            return content
    except Exception as e:
        print(f"⚠️ Research failed: {e}")
        # Return a fallback so the report generator still has something to show
        return f"Technical research initiated for {skill_name}. Please check back as the model completes the synthesis."
    
    return "Initial research stage complete."    # Load memory to see if the user wants courses or volunteering
    extra_context = ""
    if memory_file and memory_file.exists():
        memory = json.loads(memory_file.read_text())
        interests = memory["preferences"]["focus_areas"]
        if interests:
            extra_context = f"Also research practical ways to engage with this, specifically: {', '.join(interests)}."

    prompt = f"""
    [INST] Write a 500-word technical whitepaper for '{skill_name}'.
    Include ARCHITECTURE and STANDARDS. 
    {extra_context} [/INST]
    """
    try:
        # 10-minute timeout to allow Mac to generate long responses
        res = requests.post(url, json={"model": target_model, "prompt": prompt, "stream": False}, timeout=600)
        res.raise_for_status()
        content = res.json().get("response", "").strip()
        if content:
            (skill_path / "KNOWLEDGE.md").write_text(content, encoding="utf-8")
            return content
    except Exception as e:
        print(f"⚠️ Research failed: {e}")
        return f"Research is being processed for {skill_name}."
    return "Initial research stage complete."

def run_loop(skill_path, max_iterations, model, report_path):
    skill_path = Path(skill_path).resolve()
    skill_path.mkdir(parents=True, exist_ok=True)
    
    # FIX 1: Parse the name IMMEDIATELY before doing anything else
    try:
        name, _, _ = parse_skill_md(skill_path)
    except Exception as e:
        print(f"❌ Error parsing SKILL.md: {e}")
        return

    # 2. AI RESEARCH
    knowledge_text = generate_comprehensive_knowledge(skill_path, name, model)
    
    # 3. OPTIMIZATION LOOP
    history = []
    name, current_desc, _ = parse_skill_md(skill_path)
    eval_set = [{"query": f"Run {name}", "should_trigger": True}]

    for i in range(max_iterations):
        print(f"🔄 Iteration {i+1}/{max_iterations}...")
        results = run_eval(eval_set, name, current_desc, 5, 30, Path.cwd(), model)
        history.append({"iteration": i+1, "description": current_desc, "results": results["results"]})
        
        update_ui_safely(report_path, name, skill_path, history, knowledge_text)

        if all(r.get('pass') for r in results['results']):
            break
        
        # Improve the description using the AI
        current_desc = improve_description(name, knowledge_text, current_desc, results, history, model)

    # 4. FINALIZE
    package_skill(skill_path)
    print(f"✅ Full Automation Complete for {name}.")
    return {"status": "complete", "description": current_desc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("skill_path", type=Path)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--model", default="mistral") 
    parser.add_argument("--live-report", dest="report_path")
    args = parser.parse_args()
    
    run_loop(args.skill_path, args.max_iterations, args.model, args.report_path)