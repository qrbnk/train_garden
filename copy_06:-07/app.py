import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import time
from flask import redirect, url_for
import random
from flask import render_template
import markdown
from pypdf import PdfReader
import uuid
import hashlib
import subprocess
import requests
from datetime import datetime
import secrets
import logging
import chromadb
from chromadb.utils import embedding_functions
from fpdf import FPDF
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict, List

from flask import Flask, request, jsonify, send_from_directory, render_template_string, session, send_file
from flask_cors import CORS
from flask_mail import Mail, Message
from ddgs import DDGS

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
COMPANION_DIR = os.path.join(BASE_DIR, "COMPANION")
MEMORY_DIR    = os.path.join(BASE_DIR, "user_vaults")
CHROMA_PATH   = os.path.join(BASE_DIR, "chroma_db")
SKILLS_DIR    = os.path.join(BASE_DIR, "skills")
SCRIPTS_DIR   = os.path.join(BASE_DIR, "scripts")
DB_FILE = "db.json"



# Setup persistent checkpointer for the chat companion using a direct sqlite3 connection
db_path = os.path.join(MEMORY_DIR, "checkpoints.sqlite")
conn_chat = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn_chat)


# Compile agent brain

# ─────────────────────────────────────────────
# GLOBAL AUTH STORE
# ─────────────────────────────────────────────
magic_codes = {}
pending_codes = {}

# ─────────────────────────────────────────────
# 1. LLM + EMBEDDINGS
# ─────────────────────────────────────────────
llm = OllamaLLM(model="llama3.1")
embeddings = OllamaEmbeddings(model="llama3.1")

# ─────────────────────────────────────────────
# 2. VECTOR DB (long-term memory)
# ─────────────────────────────────────────────
vector_db = Chroma(
    persist_directory="./companion_memory",
    embedding_function=embeddings
)

# ─────────────────────────────────────────────
# 3. LANGGRAPH STATE
# ─────────────────────────────────────────────
class CompanionState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str

CONSULTANT_SYSTEM_PROMPT = """You are a Senior Sustainability Career Consultant.
Provide deep, multi-paragraph analysis (at least 3 paragraphs).
1. Explain the market context.
2. Analyse the user's specific situation against the role.
3. Provide an actionable roadmap.
Avoid just giving links; explain the value of the advice first."""

def retrieve_node(state: CompanionState):
    last_message = state["messages"][-1]
    user_query = last_message.content if hasattr(last_message, "content") else str(last_message)
    docs = vector_db.similarity_search(user_query, k=2)
    return {"context": "\n".join(doc.page_content for doc in docs)}

def chat_node(state: CompanionState):
    messages = [
        SystemMessage(content=CONSULTANT_SYSTEM_PROMPT),
        *state["messages"]
    ]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response)]}

def save_memory_node(state: CompanionState):
    if len(state["messages"]) < 2:
        return state
    user_msg = state["messages"][-2]
    ai_msg   = state["messages"][-1]
    user_txt = user_msg.content if hasattr(user_msg, "content") else str(user_msg)
    ai_txt   = ai_msg.content  if hasattr(ai_msg,  "content") else str(ai_msg)
    combined = f"User: {user_txt} | Assistant: {ai_txt}"
    vector_db.add_texts([combined])
    return state

# Build companion graph
builder = StateGraph(CompanionState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("chat",     chat_node)
builder.add_node("save",     save_memory_node)

builder.add_edge(START,      "retrieve")
builder.add_edge("retrieve", "chat")
builder.add_edge("chat",     "save")
builder.add_edge("save",     END)


# ─────────────────────────────────────────────
# 4. FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

logging.basicConfig(level=logging.INFO)

for d in [COMPANION_DIR, MEMORY_DIR, CHROMA_PATH, SKILLS_DIR, SCRIPTS_DIR]:
    os.makedirs(d, exist_ok=True)

def get_history(user_id):
    if not os.path.exists(DB_FILE): 
        return []
    with open(DB_FILE, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Handle empty or corrupted json file gracefully
            data = {}
    return data.get(user_id, {}).get("history", [])

def save_message(user_id, role, content):
    data = {}
    
    # Load existing data safely
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # If file is blank, start with a fresh dictionary
                data = {}

    # Initialize user if new
    if user_id not in data:
        data[user_id] = {"history": []}
    
    # Append new message
    data[user_id]["history"].append({"role": role, "content": content})
    
    # Save back to file
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ─── ChromaDB (separate from LangChain Chroma above) ───
memory_client = chromadb.PersistentClient(path=CHROMA_PATH)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
vector_memory = memory_client.get_or_create_collection(
    name="user_vaults",
    embedding_function=embed_fn
)

# ─── Email ───
app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME="dashalolvakulenko@gmail.com",
    MAIL_PASSWORD="xwfh uphs ykfq rjwc"
)
mail = Mail(app)

# ─────────────────────────────────────────────
# 5. HTML TEMPLATE
# ─────────────────────────────────────────────
# Use [[TITLE]] / [[CONTENT]] as placeholders (no f-string, so braces are safe)
HTML_TEMPLATE = """
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
        .tag-read  { background: #fef3c7; color: #92400e; }
        .tag-apply { background: #dcfce7; color: #166534; }
        .header-bar {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 30px; border-bottom: 2px solid #f1f5f9; padding-bottom: 15px;
        }
        .btn-export {
            background: #091747; color: white; border: none; padding: 10px 20px;
            border-radius: 8px; font-weight: 700; font-size: 12px; cursor: pointer;
        }
        @media print { .no-print { display: none !important; } body { padding: 0; } }
    </style>
</head>
<body>
    <div class="header-bar no-print">
        <h1>[[TITLE]]</h1>
        <button class="btn-export" onclick="window.print()">Export PDF</button>
    </div>
    <div id="roadmap-container">[[CONTENT]]</div>
    <script>
        document.querySelectorAll('li').forEach(item => {
            item.classList.add('day-card');
            let content = item.innerHTML;
            if (content.includes('LEARN')) content = '<span class="tag tag-learn">Course</span>' + content;
            if (content.includes('READ'))  content = '<span class="tag tag-read">Insight</span>' + content;
            if (content.includes('APPLY')) content = '<span class="tag tag-apply">Action</span>' + content;
            item.innerHTML = content;
        });
    </script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# 6. CORE UTILITIES
# ─────────────────────────────────────────────

def query_ai(prompt, model="mistral"):
    """Call the local Ollama endpoint."""
    OLLAMA_URL = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"Error with model {model}: {str(e)}"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() or "" for page in reader.pages)


def save_to_memory(user_id, text, source_type):
    vector_memory.add(
        documents=[text],
        metadatas=[{"user_id": user_id, "type": source_type, "date": datetime.now().isoformat()}],
        ids=[f"{user_id}_{datetime.now().timestamp()}"]
    )


def search_memory(user_id, query, category=None):
    where_clause = {"user_id": user_id}
    if category:
        where_clause["type"] = category
    results = vector_memory.query(
        query_texts=[query],
        n_results=5,
        where=where_clause
    )
    return "\n".join(results["documents"][0]) if results["documents"] else ""


def recall_relevant_context(user_input, user_id):
    if not user_input or not isinstance(user_input, str):
        return ""
    try:
        results = vector_memory.query(
            query_texts=[user_input],
            n_results=3,
            where={"user_id": user_id}
        )
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs)
    except Exception as e:
        logging.error(f"Vector memory error: {e}")
        return ""


def memorize_event(user_id, content, category, importance_score=5):
    vector_memory.add(
        documents=[content],
        metadatas=[{"user_id": user_id, "type": category, "importance": importance_score}],
        ids=[f"{user_id}_{datetime.now().timestamp()}"]
    )
    mem_file = os.path.join(MEMORY_DIR, f"{user_id}.json")
    user_data = {"history": []}
    if os.path.exists(mem_file):
        with open(mem_file, "r") as f:
            try:
                user_data = json.load(f)
            except Exception:
                pass
    user_data["history"].insert(0, {
        "name": content[:30] + "...",
        "timestamp": datetime.now().isoformat(),
        "id": str(uuid.uuid4())
    })
    with open(mem_file, "w") as f:
        json.dump(user_data, f, indent=4)


def update_user_history(user_id, report_name, report_path):
    mem_file = os.path.join(MEMORY_DIR, f"{user_id}.json")
    data = {"history": []}
    if os.path.exists(mem_file):
        with open(mem_file, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                pass
    data["history"].insert(0, {
        "name": report_name,
        "path": report_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    with open(mem_file, "w") as f:
        json.dump(data, f, indent=4)


def get_link_preview(url):
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            text = re.sub(r"<[^<]+?>", "", response.text)
            return " ".join(text.split())[:1000]
    except Exception as e:
        logging.error(f"Scrape failed for {url}: {e}")
    return "Summary unavailable."


def research_resources(query, user_id=None):
    live_results = []
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=8))
            for r in raw_results:
                url = r["href"]
                try:
                    check = requests.head(url, timeout=3, allow_redirects=True)
                    if check.status_code == 200:
                        preview = get_link_preview(url)
                        live_results.append({"url": url, "title": r["title"], "body": preview})
                except Exception:
                    continue
                if len(live_results) >= 3:
                    break
    except Exception as e:
        logging.error(f"Search failed: {e}")
    return live_results


def generate_chat_title(user_input):
    if len(user_input) < 15:
        return user_input
    prompt = f'Briefly summarize this user request into a 4-word professional title.\nRequest: "{user_input}"\nTitle:'
    return query_ai(prompt).replace('"', "").replace(".", "").strip()


# ─────────────────────────────────────────────
# 7. ROADMAP GENERATION  (single, clean version)
# ─────────────────────────────────────────────

def generate_personalized_roadmap(user_id, user_input):
    skill_topic = user_input.replace("roadmap", "").replace("for", "").strip()
    folder_name = re.sub(r"[^a-z0-9]", "-", skill_topic.lower())[:30]
    workspace   = os.path.join(SKILLS_DIR, folder_name)
    os.makedirs(workspace, exist_ok=True)

    # Search for local context
    search_context = ""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                f"volunteering opportunities and sustainability projects in London 2026 {skill_topic}",
                max_results=6
            )
            search_context = "\n".join(f"- {r['title']}: {r['href']}" for r in results)
    except Exception:
        pass

    prompt = f"""Act as a London-based Career Placement Officer.
The user wants to develop skills in: {skill_topic}.

CRITICAL: Every day must have three bullet points starting EXACTLY with these labels:
- LEARN: (What to study)
- READ:  (Industry news or specific London orgs)
- APPLY: (A specific volunteering action in London)

If the user asks a general or random question, provide a helpful answer and 3 relevant
search-based suggestions to steer them back to their career path.

Search Context:
{search_context}
"""

    roadmap_md   = query_ai(prompt, model="llama3.1")
    roadmap_html = markdown.markdown(roadmap_md)
    styled_html  = (
        HTML_TEMPLATE
        .replace("[[TITLE]]",   f"🚀 Career Accelerator: {skill_topic.title()}")
        .replace("[[CONTENT]]", roadmap_html)
    )

    file_path = os.path.join(workspace, "report.html")
    with open(file_path, "w") as f:
        f.write(styled_html)

    path_url = f"/view-skill/{folder_name}/report"
    update_user_history(user_id, f"Roadmap: {skill_topic.title()}", path_url)
    return path_url


# ─────────────────────────────────────────────
# 8. AGENTIC RESEARCH GRAPH
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[dict], add_messages]
    results:  List[dict]
    critique: str
    report_path: str


def research_node(state: AgentState):
    last = state["messages"][-1]
    query = last["content"] if isinstance(last, dict) else last.content
    raw_results = research_resources(query)
    return {"results": raw_results}


def reflection_node(state: AgentState):
    results = state.get("results", [])
    prompt  = f"Review these results: {results}. Are they sufficient for a career roadmap? Answer YES or NO."
    critique = query_ai(prompt, model="llama3.1")
    return {"critique": critique}


def should_continue(state: AgentState):
    return "improve" if "NO" in state.get("critique", "") else END


research_builder = StateGraph(AgentState)
research_builder.add_node("research", research_node)
research_builder.add_node("reflect",  reflection_node)

research_builder.add_edge(START, "research")
research_builder.add_edge("research", "reflect")
research_builder.add_conditional_edges(
    "reflect",
    should_continue,
    {"improve": "research", END: END}
)

conn_research = sqlite3.connect(db_path, check_same_thread=False)
research_memory = SqliteSaver(conn_research)

# Compiles the research graph safely
graph = research_builder.compile(checkpointer=research_memory)

# Compiles the chat companion graph using the thread-safe global 'memory' checkpointer
agent_brain = builder.compile(checkpointer=memory)


# ─────────────────────────────────────────────
# 9. ORCHESTRATOR  (single, correct class)
# ─────────────────────────────────────────────
class UnifiedOrchestrator:
    def is_link_live(self, url):
        try:
            r = requests.head(url, timeout=3, allow_redirects=True)
            return r.status_code == 200
        except Exception:
            return False

    def process(self, user_input, user_id):
        # Tie the thread directly to the verified user_id
        config = {"configurable": {"thread_id": f"research_{user_id}"}}
        final_state = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        )

        if final_state.get("results"):
            path = generate_personalized_roadmap(user_id, user_input)
            return {"message": "Roadmap generated successfully.", "path": path}

        # Conversational agent fallback tied to user_id
        agent_config = {"configurable": {"thread_id": f"chat_{user_id}"}}
        agent_state  = agent_brain.invoke(
            {"messages": [HumanMessage(content=user_input)], "context": ""},
            config=agent_config
        )
        
        last_msg = agent_state["messages"][-1]
        reply    = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        return {"message": reply, "suggestions": []}

orchestrator = UnifiedOrchestrator()


# ─────────────────────────────────────────────
# 10. FLASK ROUTES
# ─────────────────────────────────────────────




@app.route("/static/vault/<path:filename>")
def serve_vault_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static/vault"), filename)


# ── Chat ──────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat_route():
    data       = request.json or {}
    user_input = data.get("text", "").strip()
    user_id    = data.get("user_id", "anonymous")

    # Initial greeting
    if not user_input or user_input == "INITIALIZE_GARDEN_SESSION":
        greeting = (
            "Welcome to the Train Garden. I'm your sustainability career strategist. "
            "To get started, would you like to:\n"
            "1. Upload your CV for a gap analysis?\n"
            "2. Explore specific green career paths?\n"
            "3. Search for current sustainability roles?"
        )
        return jsonify({"response": greeting})

    try:
        # Save the user's incoming message to long-term flat file
        save_message(user_id, "user", user_input)

        # Process through the orchestrator
        result = orchestrator.process(user_input, user_id)
        
        response_text = result.get("message", "")
        
        # Save the AI's response to long-term flat file
        if response_text:
            save_message(user_id, "assistant", response_text)

        return jsonify({
            "response":    response_text,
            "path":        result.get("path"),
            "suggestions": result.get("suggestions", [])
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"response": f"System Error: {str(e)}"}), 500

# ── CV Upload ─────────────────────────────────
@app.route("/api/upload-cv", methods=["POST"])
def upload_cv():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"success": False, "error": "No file"}), 400

        user_id = request.form.get("user_id", "anonymous")
        job_desc = request.form.get("job_description", "")

        temp_path = os.path.join(COMPANION_DIR, f"{user_id}_cv.pdf")
        file.save(temp_path)

        cv_text = extract_text_from_pdf(temp_path)

        prompt = (
            f"Analyse this CV against: {job_desc}\n\n"
            f"CV: {cv_text[:2000]}\n\n"
            'Return Markdown report and SKILLS_JSON: {"matched": [], "gaps": []}'
        )

        analysis_raw = query_ai(prompt)

        skills_data = {"matched": [], "gaps": []}
        clean_markdown = analysis_raw

        if "SKILLS_JSON:" in analysis_raw:
            parts = analysis_raw.split("SKILLS_JSON:")
            clean_markdown = parts[0].strip()
            try:
                skills_data = json.loads(parts[1].strip())
            except Exception:
                pass

        analysis_html = markdown.markdown(
            clean_markdown,
            extensions=["tables"]
        )

        styled_html = f"""
        ...
        """

        folder_name = f"analysis-{int(time.time())}"
        workspace = os.path.join(SKILLS_DIR, folder_name)
        os.makedirs(workspace, exist_ok=True)

        with open(os.path.join(workspace, "report.html"), "w") as f:
            f.write(styled_html)

        path_url = f"/view-skill/{folder_name}/report"

        update_user_history(
            user_id,
            "CV Gap Analysis",
            path_url
        )

        return jsonify({
            "success": True,
            "message": clean_markdown,
            "skills_data": skills_data,
            "path": path_url
        })

    except Exception as e:
        logging.error(f"Upload Error: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
# ── Auth ──────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    email   = (request.json or {}).get("email", "")
    user_id = hashlib.md5(email.encode()).hexdigest()[:12]
    return jsonify({"success": True, "user_id": user_id})


@app.route("/api/auth/send-code", methods=["POST"])
def send_code():
    email = (request.json or {}).get("email")
    if not email:
        return jsonify({"success": False, "message": "Email is required"}), 400
    code               = str(random.randint(1000, 9999))
    pending_codes[email] = code
    print(f"DEBUG: Magic Code for {email} is {code}")
    return jsonify({"success": True, "message": "Code sent. Check terminal."})


@app.route("/api/auth/request", methods=["POST"])
def handle_request_magic_code():
    data  = request.json or {}
    email = data.get("email")
    if not email:
        return jsonify({"success": False, "message": "Email is required"}), 400

    code             = str(random.randint(1000, 9999))
    magic_codes[email] = code

    print(f"\n{'*'*32}")
    print(f"MAGIC CODE FOR {email}: {code}")
    print(f"{'*'*32}\n")

    try:
        msg      = Message("Your Garden Access Code",
                           sender=app.config.get("MAIL_USERNAME"),
                           recipients=[email])
        msg.body = f"Your access code is: {code}"
        mail.send(msg)
    except Exception as e:
        print(f"MAIL ERROR: {e} (code saved in memory for manual entry)")

    return jsonify({"success": True, "message": "Code generated"})


@app.route("/api/auth/verify", methods=["POST"])
def verify():
    data               = request.json or {}
    email              = data.get("email")
    user_provided_code = str(data.get("code", "")).strip()
    stored_code        = magic_codes.get(email)

    print(f"DEBUG: Stored={stored_code}  Provided={user_provided_code}")

    if stored_code and str(stored_code) == user_provided_code:
        user_id = hashlib.md5(email.encode()).hexdigest()
        return jsonify({"success": True, "user_id": user_id})

    return jsonify({"success": False, "message": "Invalid code"}), 401

# Pages
@app.route('/')
def serve_landing():
    return send_from_directory(BASE_DIR, 'code.html')

@app.route('/dashboard')
def serve_dashboard():
    return send_from_directory(BASE_DIR, 'dashboard.html')

@app.route('/chat')
def serve_chat():
    return send_from_directory(BASE_DIR, 'chatbot.html')

@app.route('/strategist')
def serve_strategist():
    return send_from_directory(BASE_DIR, 'strategist.html')

@app.route('/roadmap')
def serve_roadmap():
    return send_from_directory(BASE_DIR, 'roadmap.html')

# API Endpoints (The "Glue")
@app.route("/api/dashboard-stats")
def get_stats():
    # Return JSON: {"total_reports": 5}
    return jsonify({"total_reports": 5})

@app.route("/api/get-history", methods=["GET"])
def get_history_api():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"history": []}), 400
        
    # Fetches previous messages linked to this email's hash
    chat_history = get_history(user_id)
    return jsonify({"history": chat_history})


@app.route("/api/manage-history", methods=["POST"])
def manage_history():
    data      = request.json or {}
    user_id   = data.get("user_id")
    action    = data.get("action")
    item_path = data.get("path")

    mem_file  = os.path.join(MEMORY_DIR, f"{user_id}.json")
    user_data = {"history": []}
    if os.path.exists(mem_file):
        with open(mem_file, "r") as f:
            try:
                user_data = json.load(f)
            except Exception:
                pass

    if action == "delete":
        user_data["history"] = [h for h in user_data["history"] if h.get("path") != item_path]
    elif action == "pin":
        for item in user_data["history"]:
            if item.get("path") == item_path:
                item["pinned"] = not item.get("pinned", False)
        user_data["history"].sort(key=lambda x: x.get("pinned", False), reverse=True)

    with open(mem_file, "w") as f:
        json.dump(user_data, f, indent=4)

    return jsonify({"success": True})


@app.route("/api/dashboard-stats")
def get_dashboard_stats():
    user_id  = request.args.get("user_id")
    mem_file = os.path.join(MEMORY_DIR, f"{user_id}.json")
    if os.path.exists(mem_file):
        with open(mem_file, "r") as f:
            data    = json.load(f)
            history = data.get("history", [])
            return jsonify({"total_reports": len(history), "recent_activity": history[:3]})
    return jsonify({"total_reports": 0, "recent_activity": []})


# ── Export PDF ────────────────────────────────
@app.route("/api/export-pdf", methods=["POST"])
def export_pdf():
    try:
        data         = request.json or {}
        title        = data.get("title", "Career Report")
        content      = data.get("content", "")
        safe_content = content.encode("latin-1", "ignore").decode("latin-1")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 18)
        pdf.cell(0, 15, text=title.upper(), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(10)
        pdf.set_font("helvetica", size=11)
        pdf.multi_cell(0, 8, text=safe_content)

        output_path = os.path.abspath("latest_report.pdf")
        pdf.output(output_path)
        return send_file(output_path, as_attachment=True)

    except Exception as e:
        print(f"PDF Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Report Viewer ─────────────────────────────
@app.route("/view-skill/<skill_name>/report")
def view_report(skill_name):
    target_dir = os.path.join(SKILLS_DIR, skill_name)
    response   = send_from_directory(target_dir, "report.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.after_request
def add_header(r):
    r.headers["X-Frame-Options"] = "SAMEORIGIN"
    return r


# Keep only this one
@app.route("/")
def index():
    return redirect(url_for("dashboard"))


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)