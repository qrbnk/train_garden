#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path

def generate_html(data: dict, auto_refresh: bool = False, skill_name: str = "", skill_path: str = "") -> str:
    # --- 1. DATA EXTRACTION ---
    history = data.get("history", [])
    title_prefix = html.escape(skill_name + " — ") if skill_name else ""
    embedded_research = data.get("research_base", "")
    
    # --- 2. KNOWLEDGE CONTENT LOGIC ---
    if embedded_research:
        knowledge_content = html.escape(embedded_research).replace("\n", "<br>")
    elif skill_path:
        target_dir = Path(skill_path).resolve()
        k_file = target_dir / "KNOWLEDGE.md"
        if k_file.exists():
            raw_text = k_file.read_text(encoding="utf-8").strip()
            knowledge_content = html.escape(raw_text).replace("\n", "<br>")
        else:
            knowledge_content = "Research base not found."
    else:
        knowledge_content = "No research data available."

    # --- 3. UI STYLING (Light Mode Only) ---
    html_header = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title_prefix}Skill Report</title>
    <style>
        body {{ font-family: sans-serif; padding: 40px; line-height: 1.6; color: #1e293b; background: #fff; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 12px; text-align: left; }}
        th {{ background: #f8fafc; }}
        .pass {{ color: #059669; font-weight: bold; text-align: center; }}
        .fail {{ color: #dc2626; font-weight: bold; text-align: center; }}
        @media print {{ .no-print {{ display: none; }} }}
        .editable-field:hover {{ background: #f9f9f9; cursor: text; border: 1px dashed #ccc; }}
    </style>
</head>
<body>
    <div class="no-print" style="margin-bottom: 20px;">
        <button onclick="window.print()" style="padding: 10px 20px; background: #091747; color: white; border: none; border-radius: 5px; cursor: pointer;">
            Download as PDF
        </button>
        <span style="margin-left: 10px; color: #666; font-size: 12px;">(Click any text below to edit before saving)</span>
    </div>
    <h2>Research Knowledge Base</h2>
    <div class="editable-field" contenteditable="true" style="padding: 15px; border: 1px solid #eee; border-radius: 8px; margin-bottom: 30px;">
        {knowledge_content}
    </div>
"""

    # --- 4. TABLE GENERATION ---
    if not history:
        return html_header + "<p>No iteration history available yet.</p></body></html>"

    # Get queries from the first iteration for table headers
    first_results = history[0].get("results", [])
    train_queries = [r["query"] for r in first_results]
    
    table_html = "<h3>Iteration Logs</h3><table><thead><tr><th>Iter</th><th>Description</th>"
    for q in train_queries: 
        table_html += f"<th>{html.escape(q)}</th>"
    table_html += "</tr></thead><tbody>"

    for h in history:
        res_list = h.get("results", [])
        table_html += f"<tr><td>{h['iteration']}</td><td style='font-family: monospace; color: #005cc5;'>{html.escape(h['description'])}</td>"
        res_map = {r["query"]: r for r in res_list}
        for q in train_queries:
            r = res_map.get(q, {})
            icon, cls = ("✓", "pass") if r.get("pass") else ("✗", "fail")
            table_html += f"<td class='{cls}'>{icon}</td>"
        table_html += "</tr>"

    return html_header + table_html + "</tbody></table></body></html>"

if __name__ == "__main__":
    # Test block or pass
    pass