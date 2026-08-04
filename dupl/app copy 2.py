import os
import json
import time
import random
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
import logging
import chromadb
from chromadb.utils import embedding_functions
from weasyprint import HTML
from fpdf import FPDF
import re
from flask import Flask, request, jsonify, render_template, send_file
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)
mail = Mail(app)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPANION_DIR = os.path.join(BASE_DIR, "COMPANION")
MEMORY_DIR = os.path.join(COMPANION_DIR, "memory")
SKILLS_DIR = os.path.join(COMPANION_DIR, "skills")
SCRIPTS_DIR = os.path.join(COMPANION_DIR, "scripts")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "user_vaults") # Ensure this is consistent
# --- SINGLE SOURCE OF TRUTH ---
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
memory_client = chromadb.PersistentClient(path=CHROMA_PATH)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
pending_codes = {}

vector_memory = memory_client.get_or_create_collection(
    name="user_vaults", 
    embedding_function=embed_fn
)

# Stick to SentenceTransformer across the whole app
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")



if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)


try:
    user_vaults_collection = memory_client.get_or_create_collection(
        name="user_vaults",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    logging.info("✅ user_vaults collection is synchronized and ready.")
except Exception as e:
    logging.error(f"❌ Critical Database Error: {e}")

# This ensures the collection exists before the chat tries to use it
try:
    collection = memory_client.get_or_create_collection(name="user_vaults")
    logging.info("Successfully connected to user_logs collection")
except Exception as e:
    logging.error(f"Failed to initialize collection: {e}")


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

# --- 1. CONSOLIDATED MEMORY ENGINE ---
memory_client = chromadb.PersistentClient(path="./agent_memory")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
vector_memory = memory_client.get_or_create_collection(name="user_vaults", embedding_function=embed_fn)

def multi_search(query):
    engines = {
        "Brave": f"https://api.search.brave.com/res/v1/web/search?q={query}",
        "DuckDuckGo": f"https://api.duckduckgo.com/?q={query}&format=json",
        "Google": f"https://google.serper.dev/search"
    }
    
    results = []
    
    # Run searches in parallel for speed
    with ThreadPoolExecutor() as executor:
        # Example: Brave Search (Assuming you have an API key)
        brave_res = executor.submit(requests.get, engines["Brave"], headers={"X-Subscription-Token": "YOUR_KEY"})
        
        # Example: Google Serper (Low-cost/Fast)
        serper_res = executor.submit(requests.post, engines["Google"], 
                                    headers={"X-API-KEY": "YOUR_KEY"}, 
                                    json={"q": query})
        
        # Compile snippets from all successful responses
        results.append(brave_res.result().json().get('web', {}).get('results', []))
        results.append(serper_res.result().json().get('organic', []))
        
    return results

def save_to_memory(user_id, text, source_type):
    """Saves a snippet of info (CV text, Chat, or Research) to the vector store."""
    vector_memory.add(
        documents=[text],
        metadatas=[{"user_id": user_id, "type": source_type, "date": datetime.now().isoformat()}],
        ids=[f"{user_id}_{datetime.now().timestamp()}"]
    )

# --- 1. CONSOLIDATED MEMORY ENGINE ---
# Ensure you only have ONE of these blocks in your file

# Use 'chroma_client' consistently
chroma_client = chromadb.PersistentClient(path="./agent_memory")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# We will use 'vector_memory' as the primary name for our collection
vector_memory = chroma_client.get_or_create_collection(
    name="user_vaults", 
    embedding_function=sentence_transformer_ef
)

def search_memory(user_id, query):
    """Retrieves the most relevant 3 pieces of information the agent remembers."""
    # This now matches the 'vector_memory' name defined above
    results = vector_memory.query(
        query_texts=[query],
        n_results=3,
        where={"user_id": user_id}
    )
    return "\n".join(results['documents'][0]) if results['documents'] else ""

def get_or_create_title(conversation_id, first_message):
    # 1. Check database for existing title
    # (Replacing 'Conversation' with whatever your Model is named)
    existing_conv = Conversation.query.filter_by(id=conversation_id).first()
    
    if existing_conv and existing_conv.title and existing_conv.title != "New Chat":
        return existing_conv.title
    
    # 2. If no meaningful title exists, ask Mistral
    summary_prompt = f"""
    Briefly summarize this user request into a 3-word professional title.
    Request: "{first_message}"
    Title:"""
    
    new_title = query_ai(summary_prompt).replace('"', '').strip()
    
    # 3. Save the new title back to the DB so we don't repeat this next time
    if existing_conv:
        existing_conv.title = new_title
        db.session.commit()
        
    return new_title

def generate_chat_title(user_input):
    """Generates a 3-5 word title for the chat history sidebar."""
    # If it's a very short message, don't bother the AI
    if len(user_input) < 15:
        return user_input

    summary_prompt = f"""
    Briefly summarize this user request into a 4-word professional title.
    Request: "{user_input}"
    Title:"""
    
    # Use your existing query_ai function
    title = query_ai(summary_prompt)
    
    # Clean up any quotes or extra periods the AI might add
    return title.replace('"', '').replace('.', '').strip()

def memorize_fact(user_id, text, metadata):
    """Stores a fact in the vector database."""
    vector_memory.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"{user_id}_{datetime.now().timestamp()}"]
    )

def recall_relevant_context(query, user_id):
    """Finds the most relevant past information."""
    results = vector_memory.query(
        query_texts=[query],
        n_results=3,
        where={"user_id": user_id}
    )
    return "\n".join(results['documents'][0]) if results['documents'] else ""

# --- 1. CONSOLIDATED MEMORY ENGINE ---
memory_client = chromadb.PersistentClient(path="./agent_memory")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
vector_memory = memory_client.get_or_create_collection(name="user_vaults", embedding_function=embed_fn)

def memorize_event(user_id, content, category, title="Data Entry", path=None):
    """The single source of truth for all AI memory and UI history."""
    # AI Memory (Vector)
    vector_memory.add(
        documents=[content],
        metadatas=[{"user_id": user_id, "type": category, "title": title}],
        ids=[f"{user_id}_{datetime.now().timestamp()}"]
    )

    # 2. Save to JSON (Sidebar History) - FORCE LOGGING
    # Even if there is no file path, we record the interaction
    update_user_history(user_id, title, path or "#")

    # UI History (Sidebar)
    if path:
        update_user_history(user_id, title, path)

def aggregate_search(query):
    # This combines Brave (for privacy/web), Wikipedia (for facts), 
    # and DuckDuckGo (via a simple library)
    results = []
    
    # 1. Brave Search (Existing)
    # 2. Wikipedia (Existing)
    
    # 3. DuckDuckGo (Add this for extra coverage)
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            ddg_results = [r for r in ddgs.text(query, max_results=3)]
            results.extend(ddg_results)
    except Exception as e:
        print(f"DDG Search Error: {e}")
        
    return results

def get_link_preview(url):
    """Standalone helper to scrape 1000 chars from a link for the AI to read."""
    try:
        # Use a short timeout so the chat doesn't lag
        response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            # Strip HTML tags to get clean text
            text = re.sub(r'<[^<]+?>', '', response.text)
            # Remove extra whitespace and newlines
            clean_text = " ".join(text.split())
            return clean_text[:1000] 
    except Exception as e:
        logging.error(f"Scrape failed for {url}: {e}")
    return "Summary unavailable."

def research_resources(query):
    """Harvests links and verifies they are LIVE before sending to UI."""
    live_results = []
    
    try:
        with DDGS() as ddgs:
            # We fetch 8 results to account for the fact that some will be dead/expired
            raw_results = list(ddgs.text(query, max_results=8))
            
            for r in raw_results:
                url = r['href']
                try:
                    # 1. THE LIVE CHECK: Ping the URL with a short timeout
                    # We use a 'HEAD' request because it's faster than downloading the whole page
                    check = requests.head(url, timeout=3, allow_redirects=True, headers={'User-Agent': 'Mozilla/5.0'})
                    
                    if check.status_code == 200:
                        # 2. Scrape preview for the AI to summarize
                        preview = get_link_preview(url)
                        live_results.append({
                            "url": url,
                            "title": r['title'],
                            "body": preview 
                        })
                except Exception:
                    # If the site times out or errors, we skip it silently
                    continue

                # Stop once we have 3 solid, live resources
                if len(live_results) >= 3:
                    break
                    
    except Exception as e:
        logging.error(f"Search failed: {e}")
    
    return live_results    """Harvests links and pre-scrapes them for the AI."""
    all_results = []
    # Using DuckDuckGo for free, reliable search if Serper/Brave keys aren't set
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            for r in results:
                # INTELLIGENCE: We scrape the link now so the AI can summarize it later
                preview = get_link_preview(r['href'])
                all_results.append({
                    "url": r['href'],
                    "title": r['title'],
                    "body": preview # The AI uses this for the Overview
                })
    except Exception as e:
        logging.error(f"Search failed: {e}")
    
    return all_results
    def fetch_source(name, url, headers):
        try:
            # We use a 10s timeout per engine to keep the UI snappy
            resp = requests.post(url, headers=headers, json={"q": query}, timeout=10) if name == "google" \
                   else requests.get(f"{url}?q={query}", headers=headers, timeout=10)
            return resp.json()
        except: return None

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_source, n, u, h) for n, (u, h) in sources.items()]
        for f in futures:
            res = f.result()
            if res: all_results.append(res)
            
    return all_results

def query_ai(prompt):
    """Queries local Ollama instance with fallback for UI consistency."""
    OLLAMA_URL = "http://localhost:11434/api/generate"
    # Use a shorter timeout for quicker UI feedback
    payload = {"model": "mistral", "prompt": prompt, "stream": False}
    
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=90)
        r.raise_for_status() # Check if the request actually worked
        return r.json().get('response', "").strip()
    
    except requests.exceptions.ConnectionError:
        return """
        FIT_ITEM: Offline on AI Engine: I can't reach Ollama at localhost:11434.
        
        It looks like the local AI service isn't running. Please open the Ollama app on your Mac.
        SUGGESTIONS: Restart Ollama | Check Model Status | Help
        """
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

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

def store_communication(data):
    """Safely logs interactions to history.json without crashing on objects."""
    import json
    import os
    
    # Ensure all values are JSON-serializable strings/primitives
    clean_data = {}
    for k, v in data.items():
        if hasattr(v, 'get_json'): # If it's a Flask Response, extract data
            clean_data[k] = v.get_json()
        else:
            clean_data[k] = str(v) if not isinstance(v, (dict, list, int, float, bool, type(None))) else v

    # Use 'a' to create file if missing and append new line
    try:
        with open('history.json', 'a') as f:
            f.write(json.dumps(clean_data) + "\n")
    except Exception as e:
        print(f"Storage Error: {e}")


def extract_text_from_pdf(pdf_path):
    """Reads text from an uploaded PDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --- CORE AI WORKFLOWS ---

def trigger_research_process(user_id, user_input):
    skill_name = user_input.title()
    folder_name = user_input.lower().strip().replace(" ", "-")[:30] + "-" + datetime.now().strftime("%H%M")
    workspace = os.path.join(SKILLS_DIR, folder_name)
    os.makedirs(workspace, exist_ok=True)
    
    links_html = ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(user_input, max_results=5))
            for r in results:
                links_html += f"<div><h4>{r['title']}</h4><p>{r['body'][:180]}...</p><a href='{r['href']}'>Source</a></div>"
    except Exception as e:
        links_html = f"<p>Search unavailable: {e}</p>"

    # Create file first
    report_file = os.path.join(workspace, "report.html")
    with open(report_file, "w") as f:
        f.write(f"<html><body><h1>{skill_name}</h1>{links_html}</body></html>")

    path_url = f"/view-skill/{folder_name}/report"
    
    # NOW memorize (variables are all defined)
    memorize_event(user_id, links_html, "research", f"Research: {skill_name}", path_url)
    return path_url 
    
def get_raw_path_string(filename):
    # This ensures you have a clean string path to the file
    import os
    base_path = "static/uploads" # Adjust to your actual folder
    return os.path.join(base_path, filename)

# --- ROUTES ---

@app.route('/')
def serve_index():
    return send_from_directory(BASE_DIR, 'chatbot.html')


@app.route('/api/export-pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.json
        title = data.get('title', 'Career Report')
        content = data.get('content', '')

        # Sanitize for FPDF (removes characters it can't print)
        safe_content = content.encode('latin-1', 'ignore').decode('latin-1')

        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("helvetica", "B", 18)
        pdf.cell(0, 15, text=title.upper(), new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(10)
        
        # Body Content
        pdf.set_font("helvetica", size=11)
        pdf.multi_cell(0, 8, text=safe_content)
        
        output_path = os.path.abspath("latest_report.pdf")
        pdf.output(output_path)
        
        return send_file(output_path, as_attachment=True)
        
    except Exception as e:
        print(f"PDF Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register():
    email = request.json.get('email')
    user_id = hashlib.md5(email.encode()).hexdigest()[:12]
    return jsonify({"success": True, "user_id": user_id})



@app.route('/api/manage-history', methods=['POST'])
def manage_history():
    data = request.json
    user_id = data.get('user_id')
    action = data.get('action') 
    item_path = data.get('path')
    
    os.makedirs('user_vaults', exist_ok=True)
    mem_file = os.path.join(MEMORY_DIR, f"{user_id}.json")

    # 1. Check if the file exists
    if os.path.exists(mem_file):
        # 2. Indent the 'with' block under the 'if'
        with open(mem_file, 'r') as f:
            user_data = json.load(f)
    else:
        # 3. Align this 'else' with the 'if' above
        user_data = {"history": []}
        
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
    data = request.json
    email = data.get('email')
    if not email:
        return jsonify({"success": False, "error": "Email required"}), 400

    user_id = hashlib.md5(email.encode()).hexdigest()[:12]
    magic_token = str(secrets.randbelow(8999) + 1000)
    
    # Ensure MEMORY_DIR exists!
    os.makedirs(MEMORY_DIR, exist_ok=True)
    mem_file = os.path.join(MEMORY_DIR, f"{user_id}.json")
    
    user_data = {"email": email, "history": [], "token": magic_token}
    if os.path.exists(mem_file):
        with open(mem_file, 'r') as f:
            try:
                user_data = json.load(f)
            except: pass
    
    user_data['token'] = magic_token
    with open(mem_file, 'w') as f:
        json.dump(user_data, f)

    # --- EMAIL SENDING ---
    try:
        msg = Message(
            subject=f"🛡️ {magic_token} is your Train Garden code",
            sender=app.config.get("MAIL_USERNAME"),
            recipients=[email]
        )
        msg.body = f"Your code is: {magic_token}"
        mail.send(msg)
        return jsonify({"success": True, "user_id": user_id})
    except Exception as e:
        print(f"Mail Error: {e}")
        # Even if mail fails, return success for testing so you can see the code in console
        print(f"DEBUG: Your magic code is {magic_token}") 
        return jsonify({"success": True, "user_id": user_id})

@app.route('/api/auth/send-code', methods=['POST'])
def send_code():
    data = request.json
    email = data.get('email')
    
    if not email:
        return jsonify({"success": False, "message": "Email is required"}), 400
    
    # Generate a 4-digit code
    code = str(random.randint(1000, 9999))
    
    # This uses the global dictionary we just initialized
    pending_codes[email] = code
    
    # In a real app, you'd send an email. For now, check your terminal!
    print(f"DEBUG: Magic Code for {email} is {code}") 
    
    return jsonify({"success": True, "message": "Code sent. Check terminal."})

@app.route('/api/auth/verify', methods=['POST'])
def verify_code():
    data = request.json
    email = data.get('email')
    user_code = data.get('code')
    
    # Verify the code against our global dictionary
    if email in pending_codes and pending_codes[email] == str(user_code):
        del pending_codes[email] # Clear it after use
        user_id = hashlib.md5(email.encode()).hexdigest()[:12]
        return jsonify({
            "success": True, 
            "user_id": user_id
        })
    
    return jsonify({"success": False, "message": "Invalid code"}), 401


# Temporary in-memory store for titles (use a DB in production)
conversation_titles = {}

def check_db_for_title(conv_id):
    """Checks if a title already exists for this conversation."""
    if not conv_id:
        return None
    return conversation_titles.get(conv_id)

def save_title_to_db(conv_id, title):
    """Saves the generated title so we don't query AI again."""
    if conv_id:
        conversation_titles[conv_id] = title

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('text', '').strip()
        user_id = data.get('user_id', 'anonymous')
        
        # 1. Retrieve Memory Bridge
        vault = vector_memory.get(where={"user_id": user_id})
        user_background = " ".join(vault['documents'][-3:]) if vault['documents'] else "New User"

        # 2. THE SYSTEM PROMPT (Fixed Quotes & Logic)
        system_prompt = f"""
You are Jack, a career strategist. Use the following CONTEXT to personalize your response:
CONTEXT: {user_background}

STRICT LOGIC:
1. PERCEPTION: Reference past moves (e.g., PwC, DeepMind) if found in context.
2. REASONING: Connect user skills to trending AI/Clinical roles.
3. STATUS: Every action you take MUST start with "STATUS: " (e.g., "STATUS: Web search complete").
4. PROACTIVE: Always end with a specific suggestion (e.g., [DRAFT_EMAIL] or [SCHEDULE_PREP]).

CORE OBJECTIVES:
- UPSKILLING: Suggest high-impact learning paths for gaps.
- ROLE DISCOVERY: Find 'Mission-Driven' roles.
- TONE: Warm and familiar (e.g., 'Hey Daryna, good to see you back').

STRICT FORMATTING:
STATUS: [Description of step]
OVERVIEW: [2-sentence summary of a resource]
RESOURCE: [URL] | [Action-oriented Title]
"""

        # 3. Get AI Response
        full_query = f"{system_prompt}\n\nUser Question: {user_input}"
        chat_response = query_ai(full_query)

        # 4. Parse the response for the UI
        lines = chat_response.strip().split('\n')
        resources = []
        status_updates = []
        narrative_text = []
        current_overview = ""

        for line in lines:
            if line.startswith("STATUS:"):
                status_updates.append(line.replace("STATUS:", "").strip())
            elif line.startswith("OVERVIEW:"):
                current_overview = line.replace("OVERVIEW:", "").strip()
            elif line.startswith("RESOURCE:"):
                parts = line.replace("RESOURCE:", "").split("|")
                if len(parts) >= 2:
                    resources.append({
                        "url": parts[0].strip(),
                        "desc": parts[1].strip(),
                        "overview": current_overview
                    })
                current_overview = ""
            else:
                if line.strip(): narrative_text.append(line.strip())

        return jsonify({
            "status": "success",
            "message": "\n".join(narrative_text),
            "resources": resources,
            "status_updates": status_updates # Sent to your frontend checklist
        })

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/upload-cv', methods=['POST'])
def upload_cv():
    try:
        file = request.files.get('file')
        user_id = request.form.get('user_id')
        job_desc = request.form.get('job_description', 'General Role')
        
        if not file: 
            return jsonify({"success": False, "message": "No file"}), 400
        
        # 1. Save and Extract
        temp_path = os.path.join(COMPANION_DIR, f"{user_id}_cv.pdf")
        file.save(temp_path)
        cv_text = extract_text_from_pdf(temp_path)
        
        # 2. Extract skills for the summary (Fixes the NameError)
        skills_found = cv_text[:500] 
        
        # 3. Generate the Intelligence Report
        prompt = f"Analyze CV against Job: {job_desc}\n\nCV Text: {cv_text[:2000]}"
        analysis_md = query_ai(prompt)
        analysis_html = markdown.markdown(analysis_md, extensions=['tables'])

        # 4. Create Workspace
        folder_name = "gap-analysis-" + datetime.now().strftime("%H%M")
        workspace = os.path.join(SKILLS_DIR, folder_name)
        os.makedirs(workspace, exist_ok=True)

        # Update the prompt to ask for structured JSON skills
    prompt = f"""
    Compare this CV against the Job Description: {job_desc}
    Return the analysis in two parts:
    1. A detailed Markdown report.
    2. A section at the end formatted as:
       SKILLS_JSON: {{"matched": ["Skill A", "Skill B"], "gaps": ["Skill C", "Skill D"]}}
    """
    analysis_raw = query_ai(prompt)
    
    # Extract the JSON for the UI
    skills_data = {}
    if "SKILLS_JSON:" in analysis_raw:
        json_str = analysis_raw.split("SKILLS_JSON:")[1].strip()
        skills_data = json.loads(json_str)

    return jsonify({
        "success": True,
        "message": "STATUS: Gap analysis complete. Check the Skill Vault.",
        "skills_data": skills_data, # This goes to the right sidebar
        "suggestions": ["How can I fill these gaps?", "Find courses for Skill C"]
    })

        # 5. Apply the High-End Styling
        styled_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Inter', sans-serif; padding: 40px; color: #1e293b; line-height: 1.7; background: #fff; }}
                h1 {{ color: #091747; font-size: 24px; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 25px 0; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
                th {{ background-color: #091747; color: white; text-align: left; padding: 16px; font-size: 13px; }}
                td {{ padding: 16px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="no-print"><span style="font-size:10px; color:#64748b; font-weight:bold; text-transform:uppercase;">Intelligence Report</span></div>
            <h1>Career Gap Analysis</h1>
            {analysis_html}
        </body>
        </html>
        """
        
        with open(os.path.join(workspace, "report.html"), "w") as f:
            f.write(styled_html)

        # 6. Save to ChromaDB so the AI "remembers" the skills
        collection.add(
            documents=[f"User has expertise in: {skills_found}"],
            metadatas=[{"type": "cv_skills", "user_id": user_id}],
            ids=[f"{user_id}_cv_{time.time()}"]
        )
        
        path_url = f"/view-skill/{folder_name}/report"
        
        return jsonify({
            "success": True, 
            "status": "success", 
            "path": path_url,
            "message": "STATUS: Analysis complete! Check your workspace."
        })

    except Exception as e:
        print(f"Upload Error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

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

    "If the user asks a general or random question, provide a helpful answer and 3 relevant search-based suggestions to steer them back to their career path."

    Search Context: {search_context}
    """
    
    roadmap_md = query_ai(prompt)
    roadmap_html = markdown.markdown(roadmap_md)
    
    folder_name = f"roadmap-{datetime.now().strftime('%H%M%S')}"
    folder_name = re.sub(r'[^a-z0-9]', '-', skill_topic.lower())[:30]
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