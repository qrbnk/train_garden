import os
import json
import markdown
from pypdf import PdfReader
import uuid
import hashlib
import subprocess
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from flask_mail import Mail, Message
from ddgs import DDGS
import secrets
from flask_mail import Message, Mail

app = Flask(__name__)
CORS(app)
mail = Mail(app)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPANION_DIR = os.path.join(BASE_DIR, "COMPANION")
MEMORY_DIR = os.path.join(COMPANION_DIR, "memory")
SKILLS_DIR = os.path.join(COMPANION_DIR, "skills")
SCRIPTS_DIR = os.path.join(COMPANION_DIR, "scripts")

# Create directories if they don't exist
for d in [MEMORY_DIR, SKILLS_DIR, SCRIPTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Email Setup
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='dashalolvakulenko@gmail.com',
    MAIL_PASSWORD='xwfh uphs ykfq rjwc' 
)
mail = Mail(app)

def wrap_in_ui(content_html, title):
    """Adds the Export PDF button and styling to every report."""
    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Inter', sans-serif; padding: 40px; color: #1e293b; background: #fff; line-height: 1.7; }}
            .header-bar {{ 
                display: flex; justify-content: space-between; align-items: center; 
                margin-bottom: 30px; border-bottom: 2px solid #f1f5f9; padding-bottom: 15px; 
            }}
            h1 {{ color: #091747; font-size: 24px; margin: 0; }}
            .btn-export {{ 
                background: #091747; color: white; border: none; padding: 10px 20px; 
                border-radius: 8px; font-weight: 700; font-size: 12px; cursor: pointer; 
            }}
            @media print {{ 
                .no-print {{ display: none !important; }} 
                body {{ padding: 0; }} 
            }}
            .day-card {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 25px; border-radius: 20px; margin-bottom: 20px; list-style: none; }}
            .tag {{ font-size: 10px; font-weight: 800; padding: 4px 10px; border-radius: 6px; text-transform: uppercase; margin-right: 10px; display: inline-block; }}
            .tag-learn {{ background: #dbeafe; color: #1e40af; }}
            .tag-read {{ background: #fef3c7; color: #92400e; }}
            .tag-apply {{ background: #dcfce7; color: #166534; }}
        </style>
    </head>
    <body>
        <div class="header-bar no-print">
            <h1>{title}</h1>
            <button class="btn-export" onclick="window.print()">Export PDF</button>
        </div>
        <div id="report-body">
            {content_html}
        </div>
        <div class="link-box">
    <strong>Local Action:</strong> 
    <a href="https://www.google.com/maps/search/sustainability+volunteering+London" target="_blank" class="resource-link">
        📍 Find volunteering near me
    </a>
</div>
    </body>
    </html>
    """
# --- 1. THE ROOT ROUTE (Fixes the 404) ---
@app.route('/')
def index():
    """Serves the main chatbot interface."""
    try:
        return send_from_directory(BASE_DIR, 'chatbot.html')
    except Exception:
        return "<h1>File Not Found</h1><p>Ensure chatbot.html is in the same folder as app.py</p>", 404

# --- CORE UTILITIES ---

def query_ai(prompt):
    """Queries local Ollama instance."""
    OLLAMA_URL = "http://localhost:11434/api/generate"
    payload = {"model": "mistral", "prompt": prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        return r.json().get('response', "").strip()
    except:
        return "The AI agent is currently offline. Please check Ollama."

def update_user_history(user_id, skill_name, report_path):
    """Updates the persistent JSON history for the sidebar."""
    memory_path = os.path.join(MEMORY_DIR, f"{user_id}.json")
    data = {"history": [], "preferences": []}
    if os.path.exists(memory_path):
        with open(memory_path, 'r') as f:
            try: data = json.load(f)
            except: pass
    
    # Remove existing entry of same name to push to top
    data['history'] = [h for h in data['history'] if h['name'] != skill_name]
    
    data['history'].insert(0, {
        "name": skill_name,
        "timestamp": datetime.now().isoformat(),
        "path": report_path
    })
    with open(memory_path, 'w') as f:
        json.dump(data, f, indent=4)

def extract_text_from_pdf(pdf_path):
    """Reads text from an uploaded PDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --- CORE AI WORKFLOWS ---

def trigger_research_process(user_id, user_input):
    """Generates a research report with live web links."""
    skill_name = user_input.title()
    folder_name = user_input.lower().strip().replace(" ", "-")[:30] + "-" + datetime.now().strftime("%H%M")
    workspace = os.path.join(SKILLS_DIR, folder_name)
    os.makedirs(workspace, exist_ok=True)
    
    links_html = ""
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(user_input, max_results=5):
                links_html += f"""
                <div style="border:1px solid #e2e8f0; padding:20px; border-radius:12px; margin-bottom:15px; background:white;">
                    <h4 style="margin:0 0 10px 0; color:#091747;">{r['title']}</h4>
                    <p style="font-size:14px; color:#64748b;">{r['body'][:180]}...</p>
                    <a href="{r['href']}" target="_blank" style="display:inline-block; background:#091747; color:white; padding:8px 16px; border-radius:6px; text-decoration:none; font-size:12px; font-weight:600;">Visit Source</a>
                </div>"""
    except: links_html = "<p>Web search currently unavailable.</p>"

    full_html = f"""
    <html>
    <head><style>body {{ font-family: sans-serif; padding: 40px; background: #f8fafc; color: #1e293b; line-height: 1.6; }}</style></head>
    <body><h1>{skill_name}</h1><hr style="border:1px solid #e2e8f0; margin-bottom:20px;">{links_html}</body>
    </html>"""
    
    report_file = os.path.join(workspace, "report.html")
    with open(report_file, "w") as f:
        f.write(full_html)

    path_url = f"/view-skill/{folder_name}/report"
    update_user_history(user_id, skill_name, path_url)
    return jsonify({"status": "success", "path": path_url})

# --- ROUTES ---

@app.route('/')
def serve_index():
    return send_from_directory(BASE_DIR, 'chatbot.html')

@app.route('/api/register', methods=['POST'])
def register():
    email = request.json.get('email')
    user_id = hashlib.md5(email.encode()).hexdigest()[:12]
    return jsonify({"success": True, "user_id": user_id})

@app.route('/api/chat', methods=['POST'])
def handle_message():
    data = request.json
    user_input = data.get('text', '')
    user_id = data.get('user_id')
    
    # 1. ROADMAP REQUEST
    if "roadmap" in user_input.lower() or "curriculum" in user_input.lower():
        # This function should save an HTML file and return the FILE PATH
        file_path = generate_personalized_roadmap(user_id, user_input)
        return jsonify({"path": file_path, "status": "success"})

    # 2. DEFAULT ROUTING
    intent_prompt = f"Categorize: '{user_input}'. Needs live web links? Reply 'RESEARCH' or 'CHAT'."
    decision = query_ai(intent_prompt)

    if "RESEARCH" in decision.upper():
        file_path = trigger_research_process(user_id, user_input)
        return jsonify({"path": file_path, "status": "success"})

    # ADD THIS: Default Chat Response
    chat_response = query_ai(user_input) 
    return jsonify({"status": "success", "message": chat_response})
    
    # ... handle regular CHAT here ...
    
    # ... rest of your standard chat logic ...
@app.route('/api/manage-history', methods=['POST'])
def manage_history():
    data = request.json
    user_id = data.get('user_id')
    action = data.get('action') # 'delete' or 'pin'
    item_path = data.get('path')
    
    mem_file = os.path.join(MEMORY_DIR, f"{user_id}.json")
    if not os.path.exists(mem_file): return jsonify({"success": False})

    with open(mem_file, 'r') as f:
        user_data = json.load(f)

    if action == 'delete':
        user_data['history'] = [h for h in user_data['history'] if h['path'] != item_path]
    elif action == 'pin':
        for item in user_data['history']:
            if item['path'] == item_path:
                item['pinned'] = not item.get('pinned', False)
        # Sort so pinned items are at the top
        user_data['history'].sort(key=lambda x: x.get('pinned', False), reverse=True)

    with open(mem_file, 'w') as f:
        json.dump(user_data, f, indent=4)
    
    return jsonify({"success": True})

@app.route('/api/auth/request', methods=['POST'])
def request_auth():
    email = request.json.get('email')
    # Generate the unique User ID based on email
    user_id = hashlib.md5(email.encode()).hexdigest()[:12]
    
    # Generate 4-digit security code
    magic_token = str(secrets.randbelow(8999) + 1000)
    
    # Save token to memory vault
    mem_file = os.path.join(MEMORY_DIR, f"{user_id}.json")
    if not os.path.exists(mem_file):
        user_data = {"email": email, "history": [], "token": magic_token}
    else:
        with open(mem_file, 'r') as f:
            user_data = json.load(f)
        user_data['token'] = magic_token
        
    with open(mem_file, 'w') as f:
        json.dump(user_data, f)

    # Send the actual email using your Flask-Mail config
    try:
        msg = Message(
            subject=f"🛡️ {magic_token} is your Train Garden code",
            sender=app.config.get("MAIL_USERNAME"),
            recipients=[email]
        )
        msg.body = f"Your secure magic link code is: {magic_token}\n\nUse this to unlock your career workspace."
        mail.send(msg)
        
        return jsonify({"success": True, "user_id": user_id})
    except Exception as e:
        print(f"Mail Error: {e}")
        return jsonify({"success": False, "error": "Email delivery failed"}), 500

@app.route('/api/auth/verify', methods=['POST'])
def verify_auth():
    user_id = request.json.get('user_id')
    provided_token = request.json.get('token')
    
    mem_file = os.path.join(MEMORY_DIR, f"{user_id}.json")
    with open(mem_file, 'r') as f:
        user_data = json.load(f)
        
    if user_data.get('token') == provided_token:
        # Clear token after use for security
        user_data['token'] = None
        with open(mem_file, 'w') as f:
            json.dump(user_data, f)
        return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "Invalid Token"}), 401
    
@app.route('/api/upload-cv', methods=['POST'])
def upload_cv():
    file = request.files.get('file')
    user_id = request.form.get('user_id')
    job_desc = request.form.get('job_description', 'General Role')
    
    if not file: return jsonify({"error": "No file"}), 400
    
    temp_path = os.path.join(COMPANION_DIR, "temp_cv.pdf")
    file.save(temp_path)
    
    cv_text = extract_text_from_pdf(temp_path)
    
    # Precise Prompt for a side-by-side comparison
    prompt = f"""
    Act as a Senior Career Consultant. Analyze the following CV against the Job Description.
    
    Structure your response as follows:
    1. A Table with 3 columns: "Target Role Skill", "User's Matching Experience", and "Gap Status" (Match / Partial / Missing).
    2. A short bulleted 'Strategic Summary'.
    3. A '5-Day Learning Roadmap' to bridge the critical gaps.

    Use Markdown tables for the comparison.
    
    CV TEXT: {cv_text[:2500]}
    JOB DESC: {job_desc}
    """
    
    analysis_md = query_ai(prompt)
    # Enable 'tables' extension so Markdown renders correctly
    analysis_html = markdown.markdown(analysis_md, extensions=['tables', 'fenced_code'])
    
    folder_name = "gap-analysis-" + datetime.now().strftime("%H%M")
    workspace = os.path.join(SKILLS_DIR, folder_name)
    os.makedirs(workspace, exist_ok=True)
    
    # Enhanced UI Styling for the comparison table
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Inter', sans-serif; padding: 40px; color: #1e293b; line-height: 1.7; background: #fff; }}
            h1 {{ color: #091747; font-size: 24px; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px; }}
            h2 {{ color: #091747; font-size: 18px; margin-top: 30px; }}
            
            /* Table Styling */
            table {{ width: 100%; border-collapse: collapse; margin: 25px 0; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
            th {{ background-color: #091747; color: white; text-align: left; padding: 16px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; }}
            td {{ padding: 16px; border-bottom: 1px solid #e2e8f0; font-size: 14px; background: #fff; }}
            tr:last-child td {{ border-bottom: none; }}
            
            /* Highlighting statuses */
            td:contains('Match') {{ color: #059669; font-weight: bold; }}
            td:contains('Missing') {{ color: #dc2626; font-weight: bold; }}
            
            .roadmap-box {{ background: #f0f9ff; border: 1px solid #bae6fd; padding: 20px; border-radius: 12px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="no-print">
            <span style="font-size:10px; color:#64748b; font-weight:bold; text-transform:uppercase;">Intelligence Report</span>
        </div>
        <h1>Career Gap Analysis</h1>
        {analysis_html}
    </body>
    </html>
    """
    
    with open(os.path.join(workspace, "report.html"), "w") as f:
        f.write(styled_html)
        
    path_url = f"/view-skill/{folder_name}/report"
    update_user_history(user_id, "Career Gap Analysis", path_url)
    return jsonify({"status": "success", "path": path_url})

def generate_personalized_roadmap(user_id, user_input):
    skill_topic = user_input.replace("roadmap", "").replace("for", "").strip()
    
    # 1. Define location FIRST so the search works
    location_from_input = "London"
    search_context = ""
    
    try:
        with DDGS() as ddgs:
            search_query = f"volunteering opportunities and sustainability projects in {location_from_input} 2026"
            results = ddgs.text(search_query, max_results=6)
            search_context = "\n".join([f"- {r['title']}: {r['href']}" for r in results])
    except: 
        pass

    prompt = f"""
    Act as a London-based Career Placement Officer. 
    The user wants to volunteer to gain these skills: {skill_topic}.

    CRITICAL: Every day must have three bullet points starting EXACTLY with these labels:
    - LEARN: (What to study)
    - READ: (Industry news or specific London orgs)
    - APPLY: (A specific volunteering action in London)

    Search Context: {search_context}
    """
    
    roadmap_md = query_ai(prompt)
    roadmap_html = markdown.markdown(roadmap_md)
    
    folder_name = f"roadmap-{datetime.now().strftime('%H%M%S')}"
    workspace = os.path.join(SKILLS_DIR, folder_name)
    os.makedirs(workspace, exist_ok=True)

    # 2. Plain template (No 'f' prefix)
    html_template = """
    <html>
    <head>
        <style>
            body { font-family: 'Inter', sans-serif; padding: 40px; color: #1e293b; background: #fff; line-height: 1.7; }
            h1 { color: #091747; font-size: 24px; margin-bottom: 25px; }
            .day-card { 
                background: #f8fafc; border: 1px solid #e2e8f0; padding: 25px; 
                border-radius: 20px; margin-bottom: 20px; list-style: none;
                transition: transform 0.2s;
            }
            .day-card:hover { transform: translateY(-3px); border-color: #091747; }
            .tag { 
                font-size: 10px; font-weight: 800; padding: 4px 10px; border-radius: 6px; 
                text-transform: uppercase; margin-right: 10px; display: inline-block;
                margin-bottom: 10px;
            }
            .tag-learn { background: #dbeafe; color: #1e40af; }
            .tag-read { background: #fef3c7; color: #92400e; }
            .tag-apply { background: #dcfce7; color: #166534; }
        </style>
    </head>
    <body>
        <h1>🚀 Career Accelerator: [[TITLE]]</h1>
        <div id="roadmap-container">[[CONTENT]]</div>
        <script>
            document.querySelectorAll('li').forEach(item => {
                item.classList.add('day-card');
                let content = item.innerHTML;
                if(content.includes('LEARN')) content = '<span class="tag tag-learn">Course</span>' + content;
                if(content.includes('READ')) content = '<span class="tag tag-read">Insight</span>' + content;
                if(content.includes('APPLY')) content = '<span class="tag tag-apply">Action</span>' + content;
                item.innerHTML = content;
            });
        </script>
    </body>
    </html>
    """

    # 3. Single, clean replacement
    styled_html = html_template.replace("[[TITLE]]", skill_topic.title()).replace("[[CONTENT]]", roadmap_html)
    
    with open(os.path.join(workspace, "report.html"), "w") as f:
        f.write(styled_html)
        
    update_user_history(user_id, f"Roadmap: {skill_topic.title()}", f"/view-skill/{folder_name}/report")
    return f"/view-skill/{folder_name}/report"
    
@app.route('/api/get-history')
def get_history():
    user_id = request.args.get('user_id')
    mem_file = os.path.join(MEMORY_DIR, f"{user_id}.json")
    if os.path.exists(mem_file):
        with open(mem_file, 'r') as f: return jsonify(json.load(f))
    return jsonify({"history": []})

@app.route('/view-skill/<skill_name>/report')
def view_report(skill_name):
    return send_from_directory(os.path.join(SKILLS_DIR, skill_name), "report.html")

@app.after_request
def add_header(r):
    r.headers['X-Frame-Options'] = 'SAMEORIGIN'
    return r

if __name__ == "__main__":
    app.run(debug=True, port=5000)