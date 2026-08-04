# ─────────────────────────────────────────────────────────────────────────────
# TRAIN GARDEN — app.py  (complete rebuild)
# Memory: user_vaults/{user_id}.json is the single source of truth
# Streaming: SSE (text/event-stream) with data: prefix
# ─────────────────────────────────────────────────────────────────────────────
import os, json, time, random, uuid, hashlib, requests, secrets, logging, re, sqlite3
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus
from datetime import datetime, timedelta
from typing import Annotated, TypedDict, List

import markdown
from pypdf import PdfReader
from fpdf import FPDF
from ddgs import DDGS

from flask import (Flask, request, jsonify, send_from_directory,
                   session, send_file, Response, stream_with_context)
from flask_cors import CORS
from flask_mail import Mail, Message

import chromadb
from chromadb.utils import embedding_functions

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import OllamaLLM, OllamaEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — dirs created BEFORE any DB open
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
COMPANION_DIR = os.path.join(BASE_DIR, "COMPANION")
MEMORY_DIR    = os.path.join(BASE_DIR, "user_vaults")
CHROMA_PATH   = os.path.join(BASE_DIR, "chroma_db")
SKILLS_DIR    = os.path.join(BASE_DIR, "skills")

for d in [COMPANION_DIR, MEMORY_DIR, CHROMA_PATH, SKILLS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SQLITE SAVERS  — two separate connections
# ─────────────────────────────────────────────────────────────────────────────
_chat_db     = os.path.join(MEMORY_DIR, "chat_checkpoints.sqlite")
_research_db = os.path.join(MEMORY_DIR, "research_checkpoints.sqlite")
chat_conn     = sqlite3.connect(_chat_db,     check_same_thread=False)
research_conn = sqlite3.connect(_research_db, check_same_thread=False)
memory          = SqliteSaver(chat_conn)
research_memory = SqliteSaver(research_conn)

# ─────────────────────────────────────────────────────────────────────────────
# LLM + EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────
llm        = OllamaLLM(model="llama3.1:latest")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# LangChain Chroma (conversation summaries, per-user tagged)
vector_db = Chroma(persist_directory="./companion_memory", embedding_function=embeddings)

# ChromaDB direct (semantic memory per user)
memory_client = chromadb.PersistentClient(path=CHROMA_PATH)
embed_fn      = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
vector_memory = memory_client.get_or_create_collection(name="user_vaults", embedding_function=embed_fn)

# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC SIMILARITY HELPERS
# Reuses the same all-MiniLM-L6-v2 model that already powers vector_memory,
# so job-matching / relevance-filtering get real embedding similarity instead
# of naive lowercase substring checks — no extra model load, no extra LLM call.
#
# Embeddings are cached in a plain dict (not per-text lru_cache) so a batch
# of new/uncached strings can be embedded in ONE embed_fn call instead of one
# call per skill / gap / result — meaningfully faster once a user has 10+
# skills logged or a search returns 15-25 candidate results to filter.
# ─────────────────────────────────────────────────────────────────────────────
_embed_cache: dict = {}

def _embed_texts(texts) -> None:
    """Batch-warm the embedding cache. Safe to call with texts that are
    already cached or empty — only the genuinely new, non-empty strings get
    sent to embed_fn, in a single call."""
    to_embed, seen = [], set()
    for t in texts or []:
        key = (t or "").strip().lower()
        if key and key not in _embed_cache and key not in seen:
            to_embed.append(key)
            seen.add(key)
    if not to_embed:
        return
    try:
        vecs = embed_fn(to_embed)
        for key, vec in zip(to_embed, vecs):
            _embed_cache[key] = tuple(vec)
    except Exception as e:
        logging.error(f"Batch embedding error ({len(to_embed)} texts): {e}")

def _embed_cached(text: str):
    """Single-text lookup, warming the cache on demand if this exact text
    wasn't part of an earlier batch (keeps existing call sites working)."""
    key = (text or "").strip().lower()
    if not key:
        return None
    if key not in _embed_cache:
        _embed_texts([key])
    return _embed_cache.get(key)

def _cosine_sim(text_a: str, text_b: str) -> float:
    a, b = _embed_cached(text_a), _embed_cached(text_b)
    if a is None or b is None:
        return 0.0
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def _chunk_text(haystack: str):
    words = (haystack or "").split()
    if len(words) <= 40:
        return [haystack] if haystack else []
    chunks, step = [], 25
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + 40])
        if chunk:
            chunks.append(chunk)
    return chunks

def _max_chunk_sim(haystack: str, needle: str) -> float:
    """Highest similarity between `needle` and any window of `haystack` —
    lets a short skill phrase match a paraphrase buried in a longer posting.
    Returns the raw score (not just a threshold pass) so callers can log it
    for later threshold calibration."""
    chunks = _chunk_text(haystack)
    if not chunks or not needle:
        return 0.0
    return max((_cosine_sim(needle, c) for c in chunks), default=0.0)

def _semantic_contains(haystack: str, needle: str, threshold: float = 0.42) -> bool:
    return _max_chunk_sim(haystack, needle) >= threshold

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION LOGGING
# Appends (signal, decision) pairs to a local JSONL file so the 0.42 / 0.30
# thresholds above can be replaced with data-driven ones once there's real
# usage — see the file for the fields to plot a score histogram against.
# Best-effort only: never let logging break the request path.
# ─────────────────────────────────────────────────────────────────────────────
_CALIBRATION_LOG = os.path.join(BASE_DIR, "logs", "match_calibration.jsonl")

def _log_calibration(kind: str, payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_CALIBRATION_LOG), exist_ok=True)
        with open(_CALIBRATION_LOG, "a") as f:
            f.write(json.dumps({"ts": datetime.utcnow().isoformat(),
                                 "kind": kind, **payload}) + "\n")
    except Exception as e:
        logging.error(f"Calibration log error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)
logging.basicConfig(level=logging.INFO)

app.config.update(
    MAIL_SERVER="smtp.gmail.com", MAIL_PORT=587, MAIL_USE_TLS=True,
    MAIL_USERNAME="dashalolvakulenko@gmail.com",
    MAIL_PASSWORD="xwfh uphs ykfq rjwc"
)
mail = Mail(app)

# ─────────────────────────────────────────────────────────────────────────────
# AUTH  — disk-persisted codes with 5-min TTL
# ─────────────────────────────────────────────────────────────────────────────
AUTH_DB      = os.path.join(MEMORY_DIR, "auth_codes.json")
CODE_TTL_SEC = 300
magic_codes  = {}   # fallback in-memory

def _load_auth():
    if not os.path.exists(AUTH_DB): return {}
    try:
        with open(AUTH_DB) as f: return json.load(f)
    except: return {}

def _save_auth(d):
    with open(AUTH_DB, "w") as f: json.dump(d, f)

def store_code(email, code):
    d = _load_auth()
    d[email] = {"code": code, "expires": (datetime.now() + timedelta(seconds=CODE_TTL_SEC)).isoformat()}
    _save_auth(d)
    magic_codes[email] = code   # keep in-memory fallback

def check_code(email, code):
    d = _load_auth()
    e = d.get(email)
    if not e: return magic_codes.get(email) == str(code)
    if datetime.now() > datetime.fromisoformat(e["expires"]): return False
    return str(e["code"]) == str(code)

def consume_code(email):
    d = _load_auth(); d.pop(email, None); _save_auth(d)
    magic_codes.pop(email, None)

# ─────────────────────────────────────────────────────────────────────────────
# USER VAULT  — single source of truth: user_vaults/{user_id}.json
# ─────────────────────────────────────────────────────────────────────────────
def _vault_path(uid): return os.path.join(MEMORY_DIR, f"{uid}.json")

def load_vault(uid):
    p = _vault_path(uid)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f: return json.load(f)
        except: pass
    return {
        "user_id": uid, "email": "", "name": "",
        "first_seen": datetime.now().isoformat(),
        "last_seen":  datetime.now().isoformat(),
        "sector": "", "experience_years": "", "target_role": "",
        "goals": [], "skills": [], "skill_sources": {}, "cv_summary": "",
        "messages":      [],
        "reports":       [],
        "tracked_jobs":  [],
        "documents":     [],
        "saved_items":   [],
        "folders":       [],
        "sticky_notes":  [],
        "community_posts": [],
        "location":      "",     # saved location for local search
        "aspirations":   [],     # dream career aspirations
        "hobbies":       [],     # hobbies (mapped to transferable skills)
        "dream_job":     "",     # the dream job description
        "values":        [],     # work values (impact, flexibility, etc.) — free-text list, NOT the culture step below
        "culture_preference": "",  # company-stage/culture intake answer (its own key so it never collides with 'values' above)
        "salary_range":  "",     # expected salary
        "work_type":     "",     # remote/hybrid/onsite preference
        "limitations":   [],     # constraints (location, hours, etc.)
        "conversation_context": "",  # running narrative the agent builds
        "saved_resources":   [],   # resources the user has saved (persist across sessions)
        "hobby_skill_map":   [],   # LLM-mapped hobby -> skill associations
        "skill_validations": {},   # {skill_name: {status: validated|in_progress, cert_url: ...}}
        "profile_complete":  False,
        "onboarding_done":   False,
        "headline":       "",   # short professional headline for the profile card
        "experience":     [],   # structured work history: [{id,title,company,start_date,end_date,description,source}]
        "how_heard":      "",   # onboarding: how they found the product
        "onboarding_seen": False,  # onboarding: has the pre-chat intro flow been completed
        "intake_pending": "",   # key of the intake step whose question is awaiting THIS message's answer
        "intake_skipped": []    # steps explicitly deferred by the user (e.g. target_role via "help me explore"),
                                 # resolved without ever storing a value for that field
    }

def _recompute_profile_flags(vault: dict) -> None:
    """Keep profile_complete / onboarding_done in sync with actual data,
    instead of leaving them permanently False. 'Complete' means we know
    enough to personalise (not just a name) — a target role or dream job,
    plus some sense of where they're starting from."""
    has_direction = bool(vault.get("target_role") or vault.get("dream_job"))
    has_context   = bool(vault.get("sector") or vault.get("cv_summary") or vault.get("experience_years")
                         or vault.get("experience"))
    vault["profile_complete"] = bool(vault.get("name") and has_direction and has_context)
    vault["onboarding_done"]  = bool(has_direction or len(vault.get("messages", [])) > 2)

def save_vault(uid, vault):
    vault["last_seen"] = datetime.now().isoformat()
    _recompute_profile_flags(vault)
    with open(_vault_path(uid), "w", encoding="utf-8") as f:
        json.dump(vault, f, indent=2, ensure_ascii=False)

def log_message(uid, role, content, session_id="default", **extra):
    vault = load_vault(uid)
    entry = {"uid": uid, "role": role, "content": content, "ts": datetime.now().isoformat()}
    if extra: entry.update(extra)

    if session_id == "default":
        vault.setdefault("messages", []).append(entry)
        vault["messages"] = vault["messages"][-300:]
    else:
        sessions = vault.setdefault("chat_sessions", {})
        sess = sessions.setdefault(session_id, {
            "title": "Chat", "messages": [], "created_at": datetime.now().isoformat()
        })
        sess["messages"].append(entry)
        sess["messages"] = sess["messages"][-300:]
    save_vault(uid, vault)

def add_report(uid, name, path, report_type="report"):
    vault = load_vault(uid)
    vault.setdefault("reports", []).insert(0, {
        "id": str(uuid.uuid4()), "name": name, "path": path,
        "type": report_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "pinned": False
    })
    save_vault(uid, vault)

def track_job(uid, job):
    """Add or remove a job from tracked list."""
    vault = load_vault(uid)
    jobs  = vault.setdefault("tracked_jobs", [])
    existing = next((j for j in jobs if j.get("url") == job.get("url")), None)
    if existing:
        jobs.remove(existing)
        save_vault(uid, vault)
        return False   # removed
    job["tracked_at"] = datetime.now().isoformat()
    job["status"] = "interested"
    jobs.insert(0, job)
    save_vault(uid, vault)
    return True   # added

# ─────────────────────────────────────────────────────────────────────────────
# VECTOR MEMORY HELPERS  — always tagged with user_id
# ─────────────────────────────────────────────────────────────────────────────
def store_vector(uid, text, mem_type, importance=5):
    try:
        vector_memory.add(
            documents=[text],
            metadatas=[{"user_id": uid, "type": mem_type,
                        "importance": importance,
                        "date": datetime.now().isoformat()}],
            ids=[f"{uid}_{uuid.uuid4().hex}"]
        )
    except Exception as e:
        logging.error(f"Vector store: {e}")

def recall_vector(uid, query, n=4):
    if not query or not query.strip(): return ""
    try:
        results = vector_memory.query(
            query_texts=[query], n_results=n,
            where={"user_id": uid}
        )
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs)
    except Exception as e:
        logging.error(f"Vector recall: {e}")
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH NODES
# ─────────────────────────────────────────────────────────────────────────────
class CompanionState(TypedDict):
    messages: Annotated[list, add_messages]
    context:  str
    user_id:  str

SYSTEM_PROMPT = """You are a Senior Sustainability Career Consultant for the London green economy.
You have deep knowledge of UK sustainability, ESG, net-zero policy, circular economy, and green finance.
Always give rich, actionable advice (3+ paragraphs). Personalise using any recalled memory.
When referencing organisations or roles, format as Markdown links: [Name](URL)."""

def retrieve_node(state):
    uid   = state.get("user_id", "anon")
    query = str(state["messages"][-1].content) if hasattr(state["messages"][-1], "content") else ""
    return {"context": recall_vector(uid, query)}

def chat_node(state):
    uid     = state.get("user_id", "anon")
    context = state.get("context", "")
    sys     = SYSTEM_PROMPT
    if context: sys += f"\n\n--- USER MEMORY ---\n{context}\n---"
    lines   = []
    for m in state["messages"]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    prompt   = sys + "\n\n" + "\n".join(lines) + "\nAssistant:"
    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response)]}

def save_memory_node(state):
    uid  = state.get("user_id", "anon")
    msgs = state["messages"]
    if len(msgs) < 2: return state
    u = msgs[-2].content if hasattr(msgs[-2], "content") else str(msgs[-2])
    a = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
    store_vector(uid, f"User: {u} | Assistant: {a}", "conversation", importance=5)
    try:
        vector_db.add_texts([f"User: {u} | Assistant: {a}"],
                            metadatas=[{"user_id": uid}])
    except: pass
    return state

builder = StateGraph(CompanionState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("chat",     chat_node)
builder.add_node("save",     save_memory_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "chat")
builder.add_edge("chat", "save")
builder.add_edge("save", END)
agent_brain = builder.compile(checkpointer=memory)

# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH GRAPH
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[dict], add_messages]
    results:  List[dict]
    critique: str

def ddgs_search(query, n=6):
    out = []
    try:
        with DDGS() as d:
            for r in d.text(query, max_results=n):
                out.append({"url": r.get("href",""), "title": r.get("title",""),
                            "body": r.get("body","")[:400]})
                if len(out) >= 4: break
    except Exception as e:
        logging.error(f"DDGS: {e}")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# LINK VERIFICATION — never publish a resource we haven't confirmed is live.
# Fix #1 / #4 / #7.
# ─────────────────────────────────────────────────────────────────────────────
_URL_CACHE = {}          # url -> (is_valid, checked_at)
_URL_CACHE_TTL = 3600     # re-check a URL at most once an hour

def _is_valid_url(url: str) -> bool:
    """Confirm a URL actually resolves before we ever show it to a user."""
    if not url: return False
    now = time.time()
    cached = _URL_CACHE.get(url)
    if cached and (now - cached[1]) < _URL_CACHE_TTL:
        return cached[0]
    ok = False
    headers = {"User-Agent": "Mozilla/5.0 (TrainGardenBot link-check)"}
    try:
        r = requests.head(url, timeout=6, allow_redirects=True, headers=headers)
        if r.status_code < 400:
            ok = True
        elif r.status_code in (403, 405):
            # Some sites reject HEAD — retry lightly with GET before giving up.
            r2 = requests.get(url, timeout=6, allow_redirects=True, headers=headers, stream=True)
            ok = r2.status_code < 400
    except Exception:
        ok = False
    _URL_CACHE[url] = (ok, now)
    return ok

def google_search_fallback(query: str) -> dict:
    """When we can't find/confirm a real resource, never invent a dead link —
    hand the user a live Google search instead so they can find one themselves."""
    return {
        "url":   f"https://www.google.com/search?q={quote_plus(query)}",
        "title": f"Search Google for: {query}",
        "body":  "We couldn't confirm a specific live resource for this yet — "
                 "here's a search to help you find current options yourself.",
        "type":  "search_fallback",
        "verified": True,
    }

def verify_resources(results: list, fallback_query: str, max_check: int = 12, keep: int = 8) -> list:
    """Concurrently HEAD-check candidate resources and drop dead links.
    If nothing survives, return a single Google-search fallback rather than
    publishing an unverified/broken URL."""
    candidates = [r for r in results if r.get("url")][:max_check]
    if not candidates:
        return [google_search_fallback(fallback_query)]

    verified = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_is_valid_url, r["url"]): r for r in candidates}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                if fut.result():
                    r["verified"] = True
                    verified.append(r)
            except Exception:
                continue

    if not verified:
        return [google_search_fallback(fallback_query)]

    # Preserve original relevance ordering among the ones that verified.
    order = {id(r): i for i, r in enumerate(candidates)}
    verified.sort(key=lambda r: order.get(id(r), 999))
    return verified[:keep]

_STALE_YEAR_RE = re.compile(r"\b(2019|2020|2021|2022|2023|2024)\b")
_FRESH_YEAR_RE = re.compile(r"\b(2025|2026|2027)\b")

def _is_fresh(r: dict) -> bool:
    """Reject results that are explicitly dated to a stale year with no
    indication of being current (Fix #4 — no more 2024-dated resources)."""
    text = (r.get("title","") + " " + r.get("body",""))
    stale = _STALE_YEAR_RE.search(text)
    fresh = _FRESH_YEAR_RE.search(text)
    if stale and not fresh:
        return False
    return True

def research_node(state):
    last  = state["messages"][-1]
    query = last["content"] if isinstance(last, dict) else last.content
    return {"results": ddgs_search(query)}

def reflect_node(state):
    n       = len(state.get("results", []))
    prompt  = f"Are {n} search results enough for a career answer? YES or NO only."
    reply   = query_ai(prompt)
    return {"critique": reply}

def should_continue(state):
    return "improve" if state.get("critique","").strip().upper().startswith("NO") else END

r_builder = StateGraph(AgentState)
r_builder.add_node("research", research_node)
r_builder.add_node("reflect",  reflect_node)
r_builder.add_edge(START, "research")
r_builder.add_edge("research", "reflect")
r_builder.add_conditional_edges("reflect", should_continue,
                                {"improve": "research", END: END})
graph = r_builder.compile(checkpointer=research_memory)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
_AI_DOWN_FALLBACK = (
    "I'm having trouble reaching my thinking engine right now, so I can't respond "
    "properly this second — mind trying again in a moment? (If this keeps happening, "
    "the local AI model server may need to be started.)"
)

def query_ai(prompt, model="llama3.1:latest"):
    """Call the local Ollama model. On failure, NEVER return the raw exception
    text — every call site (chat replies, profile-extraction JSON parsing,
    suggestion generation) treats whatever comes back as either the user-facing
    message or something to attempt json.loads() on. A leaked "[AI error: ...]"
    string was previously both shown directly to users as the chat reply AND
    silently defeated every downstream extraction (json.loads on it just fails,
    which is fine) — but users had no idea *why* nothing in their profile was
    filling in, because the real cause (Ollama not reachable) was hidden inside
    what looked like a normal-ish bot message instead of a clear status."""
    try:
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": model, "prompt": prompt, "stream": False},
                          timeout=300)
        r.raise_for_status()
        return _strip_meta_commentary(r.json().get("response","").strip())
    except Exception as e:
        logging.error(f"query_ai failed (model={model}, Ollama unreachable?): {e}")
        return _AI_DOWN_FALLBACK

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SANITIZATION
# Local models (llama3.1 here) sometimes tack on out-of-character meta-commentary
# after the actual answer — a "---" divider followed by a note like
# "(Note: I've followed the rules, used a new opening phrase, etc.)". This is the
# model narrating its own compliance with the system prompt, and it must NEVER
# reach the user. Strip everything from the first such marker onward.
# ─────────────────────────────────────────────────────────────────────────────
_META_COMMENTARY_PATTERNS = [
    r"\n-{3,}\s*\n?\s*\(?\s*note\s*:",           # "\n---\n(Note:" / "\n---\nNote:"
    r"\(note\s*:\s*i(?:'|’)?ve\s+followed",       # "(Note: I've followed..."
    r"\[note\s*:",
    r"\(i(?:'|’)?ve\s+followed\s+the\s+rules",
    r"\n-{3,}\s*$",                                 # trailing bare divider
]

def _strip_meta_commentary(text: str) -> str:
    """Remove any self-referential 'I followed the rules/instructions' commentary
    a model appends after its real answer. Never let the user see this."""
    if not text:
        return text
    cut = None
    for pat in _META_COMMENTARY_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m and (cut is None or m.start() < cut):
            cut = m.start()
    return text[:cut].rstrip() if cut is not None else text.strip()

def _safe_str_list(items) -> list:
    """Coerce a list of arbitrary values (as returned by LLM JSON extraction,
    which sometimes yields dicts like {"value": "impact"} instead of plain
    strings) into a clean list of non-empty strings. Used both when merging
    freshly-extracted fields into the vault AND defensively at every
    ', '.join(...) call site, since vaults saved before this fix may already
    contain non-string entries."""
    out = []
    if not items:
        return out
    for it in items:
        if it is None:
            continue
        if isinstance(it, str):
            s = it.strip()
        elif isinstance(it, dict):
            s = str(
                it.get("value") or it.get("name") or it.get("hobby")
                or it.get("skill") or it.get("goal") or it.get("label")
                or next(iter(it.values()), "")
            ).strip()
        else:
            s = str(it).strip()
        if s:
            out.append(s)
    return out

_EXP_PLACEHOLDER_DATES = {"", "unknown", "n/a", "tbd", "none"}

def _upsert_experience(vault: dict, entries) -> bool:
    """Merge structured work-history entries (title/company/dates/description)
    extracted from chat or a CV into vault['experience'].

    Matches by company name so a role mentioned once (e.g. "stealth startup,
    AI transformation work") gets refined in place — with real dates and a
    fuller description — as the person shares more about it in later turns,
    instead of spawning a duplicate entry every time it comes up again.
    A role with no company named yet is matched to the single existing
    company-less entry, if there is exactly one, for the same reason."""
    if not entries:
        return False
    changed = False
    exp = vault.setdefault("experience", [])
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        title   = str(raw.get("title") or "").strip()
        company = str(raw.get("company") or "").strip()
        start   = str(raw.get("start_date") or "").strip()
        end     = str(raw.get("end_date") or "").strip()
        desc    = str(raw.get("description") or "").strip()
        if not (title or company or desc):
            continue

        match = None
        if company:
            for e in exp:
                if e.get("company", "").strip().lower() == company.lower():
                    match = e
                    break
        if match is None and not company:
            unnamed = [e for e in exp if not e.get("company")]
            if len(unnamed) == 1:
                match = unnamed[0]

        if match is None:
            exp.append({
                "id": uuid.uuid4().hex[:8],
                "title": title, "company": company,
                "start_date": start,
                "end_date": end or ("Present" if start else ""),
                "description": desc, "source": "chat",
            })
            changed = True
            continue

        if title and len(title) > len(match.get("title", "")):
            match["title"] = title; changed = True
        if company and not match.get("company"):
            match["company"] = company; changed = True
        if start and match.get("start_date", "").strip().lower() in _EXP_PLACEHOLDER_DATES:
            match["start_date"] = start; changed = True
        cur_end = match.get("end_date", "").strip().lower()
        if end and (cur_end in _EXP_PLACEHOLDER_DATES or (cur_end == "present" and end.lower() != "present")):
            match["end_date"] = end; changed = True
        if desc and len(desc) > len(match.get("description", "")):
            match["description"] = desc; changed = True
    return changed

def _attribute_skill(vault: dict, skill: str, source_type: str, detail: str) -> bool:
    """Record where a skill came from — the CV, a chat message describing
    work/experience, or the agent's own reasoning about a hobby/experience
    (source_type='inferred', with `detail` carrying what it reasoned from).
    A skill can accumulate several sources over time; returns True only if
    this is a genuinely new source note (not a repeat of one already on
    file), so callers know whether anything actually needs saving."""
    sources = vault.setdefault("skill_sources", {})
    entry = {"type": source_type, "detail": (detail or "").strip()[:240],
              "ts": datetime.now().isoformat()}
    existing = sources.setdefault(skill, [])
    if any(e.get("type") == entry["type"] and e.get("detail") == entry["detail"] for e in existing):
        return False
    existing.append(entry)
    return True

def _add_skill(vault: dict, skill: str, source_type: str, detail: str) -> bool:
    """Add a skill to the vault's main skill list (if not already there) and
    always record/accumulate its source attribution — this is what powers
    ATS-style 'where did this skill come from' display on the profile card,
    for skills named outright (CV, chat) as well as ones the agent inferred
    from a hobby or a described experience."""
    skill = (skill or "").strip()
    if not skill:
        return False
    is_new = skill not in vault.get("skills", [])
    if is_new:
        vault.setdefault("skills", []).append(skill)
    got_new_source = _attribute_skill(vault, skill, source_type, detail)
    return is_new or got_new_source

def extract_pdf_text(path):
    return "".join(p.extract_text() or "" for p in PdfReader(path).pages)

def _sse(data): return f"data: {json.dumps(data)}\n\n"

_RESOURCE_TYPE_LABELS = {
    "course": "Course", "volunteering": "Volunteering", "event": "Event",
    "webinar": "Webinar", "community": "Community group", "job": "Job",
    "search_fallback": "Search", "resource": "Resource",
}

def build_resource_cards(resources: list) -> list:
    """Turn verified resource results into stable card objects for the
    frontend — this is what actually gets rendered as a clickable/saveable
    link, instead of relying on the LLM to type out a correct Markdown link
    inline (the #1 source of incorrect/copy-paste-only links)."""
    cards = []
    for r in resources or []:
        if not r.get("url"):
            continue
        cards.append({
            "id":    str(uuid.uuid4()),
            "title": r.get("title") or r.get("url"),
            "url":   r["url"],
            "body":  (r.get("body") or "")[:220],
            "type":  r.get("type", "resource"),
            "type_label": _RESOURCE_TYPE_LABELS.get(r.get("type","resource"), "Resource"),
        })
    return cards


# ─────────────────────────────────────────────────────────────────────────────
# JOB CARD BUILDER  — structured card data for frontend rendering
# ─────────────────────────────────────────────────────────────────────────────
def _score_job_match(text: str, vault: dict) -> dict:
    """Hybrid match score: a skill/gap counts as present if it's either a
    literal keyword hit (cheap, zero-ambiguity signal — if the posting
    literally says 'Python', that shouldn't hinge on an embedding threshold)
    or a semantic match via cosine similarity (all-MiniLM-L6-v2, same model
    as vector_memory) — so a posting that says 'liaising with community
    partners' still registers against a 'stakeholder engagement' skill with
    zero shared words. All skill/gap/posting-chunk strings for this one job
    are embedded in a single batched call rather than one call each.
    Missing-skill signals are logged (score + decision) to a local file so
    the similarity threshold can be tuned against real data later — see
    logs/match_calibration.jsonl."""
    known = [s for s in (vault or {}).get("skills", []) if s]
    gaps  = (vault or {}).get("journey", {}).get("skills_gaps", [])
    if not known and not gaps:
        return {"match_pct": None, "match_label": "", "missing_skills": []}

    gap_skills = [g.get("skill", "") for g in gaps
                  if g.get("skill") and g.get("priority", "medium") in ("high", "medium")]

    # One batched embed_fn call for everything this job needs, instead of
    # one call per skill/gap/chunk.
    _embed_texts(known + gap_skills + _chunk_text(text))

    lower_text = text.lower()

    matched = []
    for s in known:
        if s.lower() in lower_text or _semantic_contains(text, s):
            matched.append(s)

    missing = []
    for s in gap_skills:
        keyword_hit = s.lower() in lower_text
        sim_score   = _max_chunk_sim(text, s)
        is_missing  = keyword_hit or sim_score >= 0.42
        _log_calibration("gap_signal", {"skill": s, "keyword_hit": keyword_hit,
                                         "sim_score": round(sim_score, 3),
                                         "decision": is_missing})
        if is_missing:
            missing.append(s)
    missing = list(dict.fromkeys(missing))[:3]

    pct = max(30, min(95, 60 + 8 * len(matched) - 15 * len(missing)))
    if pct >= 75:
        label = "Strong match"
    elif pct >= 50:
        label = "Good stretch"
    else:
        label = f"Bigger stretch — {len(missing)} gap(s) to close" if missing else "Stretch role"
    return {"match_pct": pct, "match_label": label, "missing_skills": missing}

def build_job_card(result, user_profile=None):
    """Turn a DDGS result into a structured job card dict."""
    title   = result.get("title","Role")
    url     = result.get("url","")
    snippet = result.get("body","")

    # Try to extract company from title (common pattern: "Title at Company")
    company = ""
    if " at " in title:
        parts   = title.split(" at ", 1)
        title   = parts[0].strip()
        company = parts[1].strip()
    elif " - " in title:
        parts   = title.split(" - ", 1)
        title   = parts[0].strip()
        company = parts[1].strip()
    elif " | " in title:
        parts   = title.split(" | ", 1)
        title   = parts[0].strip()
        company = parts[1].strip()

    # Try to detect salary in snippet
    salary = ""
    sal_match = re.search(r'£[\d,]+[kK]?\s*[-–]\s*£[\d,]+[kK]?', snippet)
    if sal_match: salary = sal_match.group(0)

    # Location heuristic
    location = "London"
    for loc in ["Remote", "Hybrid", "London", "Manchester", "Birmingham"]:
        if loc.lower() in snippet.lower() or loc.lower() in title.lower():
            location = loc
            break

    # Personalised bullet points
    bullets = []
    if user_profile:
        cv = user_profile.get("cv_summary","")
        if cv:
            prompt = (f"CV: {cv[:200]}\nJob: {title} at {company}\nSnippet: {snippet[:300]}\n"
                      "Write 3 short bullet points (max 12 words each) explaining why this job "
                      "fits this person. Start each with a verb. Return only the 3 bullets, one per line.")
            raw = query_ai(prompt)
            bullets = [l.strip().lstrip("-•").strip()
                       for l in raw.split("\n") if l.strip()][:3]
    if not bullets:
        # Fallback: extract sentences from snippet
        sentences = [s.strip() for s in snippet.split(".") if len(s.strip()) > 20]
        bullets = sentences[:3]

    match = _score_job_match(f"{title} {snippet}", user_profile)

    return {
        "type":     "job_card",
        "title":    title,
        "company":  company,
        "url":      url,
        "location": location,
        "salary":   salary,
        "posted":   "",
        "bullets":  bullets,
        "snippet":  snippet[:300],
        "match_pct":       match["match_pct"],
        "match_label":     match["match_label"],
        "missing_skills":  match["missing_skills"],
    }

def _parse_salary_range(s: str):
    """Parse a '£35k - £45k' / '£35,000-£45,000' string into (low, high) in £k."""
    nums = re.findall(r'£\s*([\d,]+)\s*([kK]?)', s or "")
    vals = []
    for num, k in nums:
        n = float(num.replace(",", ""))
        if not k:
            n = n / 1000  # raw pounds -> £k
        vals.append(n)
    if len(vals) >= 2:
        return (min(vals), max(vals))
    if len(vals) == 1:
        return (vals[0], vals[0])
    return None

def salary_reality_check(job_cards: list) -> dict:
    """Aggregate salary figures already extracted from live postings into an
    honest 'here's what this actually pays right now' range, instead of the
    user finding out only after several applications."""
    lows, highs, n = [], [], 0
    for c in job_cards:
        parsed = _parse_salary_range(c.get("salary", ""))
        if parsed:
            lows.append(parsed[0]); highs.append(parsed[1]); n += 1
    if n < 2:
        return {}
    lo, hi = round(min(lows)), round(max(highs))
    return {
        "count": n, "low_k": lo, "high_k": hi,
        "summary": f"Based on {n} live postings, typical pay for this role right now is roughly £{lo}k–£{hi}k."
    }

# ─────────────────────────────────────────────────────────────────────────────
# ROADMAP REPORT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
HTML_TMPL = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>
body{font-family:'Inter',sans-serif;padding:40px;color:#1e293b;background:#fff;line-height:1.7;max-width:860px;margin:0 auto}
h1{color:#005d42;font-size:22px;border-bottom:2px solid #005d42;padding-bottom:10px;margin-bottom:25px}
h2,h3{color:#005d42} a{color:#005d42}
.day-card{background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid #005d42;padding:20px;margin-bottom:16px;list-style:none}
.tag{font-size:10px;font-weight:800;padding:3px 8px;text-transform:uppercase;margin-right:8px;display:inline-block;margin-bottom:8px}
.tag-learn{background:#dbeafe;color:#1e40af}.tag-read{background:#fef3c7;color:#92400e}.tag-apply{background:#dcfce7;color:#166534}
.btn{background:#005d42;color:white;border:none;padding:10px 20px;font-weight:700;font-size:12px;cursor:pointer;text-transform:uppercase;letter-spacing:.05em}
</style></head><body>
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:30px;border-bottom:2px solid #f1f5f9;padding-bottom:15px">
<h1>[[TITLE]]</h1><button class="btn" onclick="window.print()">Export PDF</button></div>
<div>[[CONTENT]]</div>
<script>document.querySelectorAll('li').forEach(i=>{i.classList.add('day-card');let c=i.innerHTML;
if(c.includes('LEARN'))c='<span class="tag tag-learn">Learn</span>'+c;
if(c.includes('READ'))c='<span class="tag tag-read">Read</span>'+c;
if(c.includes('APPLY'))c='<span class="tag tag-apply">Apply</span>'+c;
i.innerHTML=c;});</script></body></html>"""

ROADMAP_TMPL = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>
body{font-family:'Inter',sans-serif;padding:40px;color:#1e293b;background:#fff;line-height:1.6;max-width:900px;margin:0 auto}
h1{color:#005d42;font-size:24px;border-bottom:2px solid #005d42;padding-bottom:10px;margin-bottom:8px}
.sub{color:#475569;margin-bottom:25px;font-size:14px}
.summary{background:#f0fdf4;border-left:4px solid #005d42;padding:16px 20px;margin-bottom:30px;font-size:14px}
.phase{border:1px solid #e2e8f0;border-radius:10px;margin-bottom:22px;overflow:hidden}
.phase-head{background:#005d42;color:#fff;padding:14px 20px;display:flex;justify-content:space-between;align-items:center}
.phase-head h2{margin:0;font-size:16px;color:#fff}
.phase-dur{font-size:11px;text-transform:uppercase;letter-spacing:.05em;opacity:.85}
.phase-goal{padding:14px 20px;background:#f8fafc;font-size:13px;color:#334155;border-bottom:1px solid #e2e8f0}
.milestone{padding:14px 20px;border-bottom:1px solid #f1f5f9;display:flex;gap:12px}
.milestone:last-child{border-bottom:none}
.milestone input{margin-top:3px}
.m-title{font-weight:700;font-size:14px;margin-bottom:3px}
.m-desc{font-size:13px;color:#475569;margin-bottom:6px}
.m-skill{font-size:10px;font-weight:800;text-transform:uppercase;background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:4px;margin-right:6px}
.m-res a{display:inline-block;font-size:12px;color:#005d42;margin-right:12px;text-decoration:underline}
.btn{background:#005d42;color:white;border:none;padding:10px 20px;font-weight:700;font-size:12px;cursor:pointer;text-transform:uppercase;letter-spacing:.05em}
</style></head><body>
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
<h1>[[TITLE]]</h1><button class="btn" onclick="window.print()">Export PDF</button></div>
<div class="sub">[[SUBTITLE]]</div>
<div class="summary">[[SUMMARY]]</div>
[[PHASES]]
</body></html>"""

def _render_roadmap_html(roadmap: dict, target: str) -> str:
    """Render the structured roadmap JSON into the phase-based HTML report."""
    phases_html = []
    for i, phase in enumerate(roadmap.get("phases", []), start=1):
        milestones_html = []
        for m in phase.get("milestones", []):
            skill_tag = f'<span class="m-skill">{m.get("skill_related","")}</span>' if m.get("skill_related") else ""
            res_links = " ".join(
                f'<a href="{r.get("url","")}" target="_blank">{r.get("title","Resource")}</a>'
                for r in m.get("resources", []) if r.get("url")
            )
            milestones_html.append(f"""<div class="milestone">
<input type="checkbox" disabled>
<div><div class="m-title">{skill_tag}{m.get("title","")}</div>
<div class="m-desc">{m.get("description","")}</div>
<div class="m-res">{res_links}</div></div></div>""")
        phases_html.append(f"""<div class="phase">
<div class="phase-head"><h2>Phase {i}: {phase.get("title","")}</h2>
<span class="phase-dur">{phase.get("duration_weeks","?")} weeks</span></div>
<div class="phase-goal"><strong>Goal:</strong> {phase.get("goal","")}</div>
{"".join(milestones_html)}
</div>""")
    subtitle = f"Personalised plan toward: {target}" if target else "Your personalised sustainability career plan"
    return (ROADMAP_TMPL
            .replace("[[TITLE]]", f"Career Roadmap: {target.title() if target else 'Sustainability Career'}")
            .replace("[[SUBTITLE]]", subtitle)
            .replace("[[SUMMARY]]", roadmap.get("narrative_summary",""))
            .replace("[[PHASES]]", "\n".join(phases_html)))

ROADMAP_PROMPT = """You are a senior London sustainability career-transition consultant building a
structured, trackable career roadmap. Do NOT invent resource links — leave resources out, real
ones are attached separately. Do NOT invent facts about the person beyond what's given below.

Person's profile:
- Current background: {current_role}
- Target role: {target_role}
- CV summary: {cv_summary}
- Known skills: {skills}
- Location: {location}
- Work preference: {work_type}
- Salary expectation: {salary_range}
- Constraints: {limitations}

Skills gap assessment already completed for this person:
{gaps_summary}

Build a phased roadmap with 3-5 phases that logically build on each other (e.g. Foundation &
Orientation -> Skill-Building -> Practical Experience -> Applications & Networking -> Interview-Ready).
Each phase should be 2-6 weeks. Ground every milestone in this person's ACTUAL gaps and background —
do not give generic advice that could apply to anyone.

Return ONLY valid JSON, no other text:
{{
  "narrative_summary": "2-3 sentences, personal and specific to this person's transition",
  "total_estimated_weeks": 16,
  "phases": [
    {{
      "title": "...",
      "duration_weeks": 3,
      "goal": "one sentence: what should be true by the end of this phase",
      "milestones": [
        {{"title": "...", "description": "...", "skill_related": "must match a skill name from the gap list above, or empty string", "effort": "low|medium|high"}}
      ]
    }}
  ]
}}"""

def _seed_roadmap_milestones(uid: str, vault: dict, roadmap: dict):
    """Push each roadmap milestone into the SAME journey.milestones list used
    by the existing verify-milestone / skill-validation system, so the plan
    is actually trackable and reportable — not just a static document."""
    journey = vault.setdefault("journey", dict(JOURNEY_DEFAULTS))
    existing_titles = {m.get("title") for m in journey.get("milestones", [])}
    for i, phase in enumerate(roadmap.get("phases", []), start=1):
        for m in phase.get("milestones", []):
            title = m.get("title", "")
            if not title or title in existing_titles:
                continue
            journey.setdefault("milestones", []).append({
                "id":       str(uuid.uuid4()),
                "title":    title,
                "description": m.get("description", ""),
                "type":     "roadmap_step",
                "phase":    i,
                "phase_title": phase.get("title", ""),
                "skill_related": m.get("skill_related", ""),
                "effort":   m.get("effort", "medium"),
                "icon":     "🗺️",
                "xp":       XP_REWARDS["milestone_custom"],
                "earned_at": "",
                "verification_status": "not_started",
                "verified": False,
                "evidence": "",
            })
            existing_titles.add(title)
    vault["journey"] = journey

def gen_roadmap(uid, user_input):
    """Build a phased, milestone-based roadmap grounded in the person's real
    skills-gap assessment and real, verified resources — then wire it into
    the same journey/milestone/verification system the rest of the app uses,
    so it can actually be tracked and reported on over time (not just a
    static document)."""
    vault  = load_vault(uid)
    target = vault.get("target_role", "")
    current_role = vault.get("sector", "")
    location = vault.get("location", "") or "London"

    # 1. Make sure we have a real skills-gap assessment to ground the plan in.
    journey = vault.get("journey", {})
    if not journey.get("assessed") or not journey.get("skills_gaps"):
        assess_skills_gap(uid, current_role, target)
        vault = load_vault(uid)
        journey = vault.get("journey", {})

    gaps = journey.get("skills_gaps", [])
    gaps_summary = "\n".join(
        f"- {g.get('skill','')}: currently level {g.get('current_level',0)}/5, "
        f"need level {g.get('target_level',0)}/5 ({g.get('priority','medium')} priority)"
        for g in gaps
    ) or "No specific gaps identified yet — build a plan from general best practice for this transition."

    prompt = ROADMAP_PROMPT.format(
        current_role  = current_role or "not specified",
        target_role   = target or "a sustainability role",
        cv_summary    = vault.get("cv_summary", "no CV uploaded")[:400],
        skills        = ", ".join(_safe_str_list(vault.get("skills"))) or "not listed",
        location      = location,
        work_type     = vault.get("work_type", "no preference stated"),
        salary_range  = vault.get("salary_range", "not stated"),
        limitations   = ", ".join(_safe_str_list(vault.get("limitations"))) or "none stated",
        gaps_summary  = gaps_summary,
    )

    raw = query_ai(prompt)
    try:
        roadmap = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        assert roadmap.get("phases")
    except Exception:
        # One retry with a stricter nudge before falling back.
        raw2 = query_ai(prompt + "\n\nReturn ONLY the JSON object, nothing else.")
        try:
            roadmap = json.loads(raw2[raw2.find("{"):raw2.rfind("}")+1])
            assert roadmap.get("phases")
        except Exception:
            roadmap = {
                "narrative_summary": "We had trouble generating a fully personalised plan — "
                                     "here's a general structure to start from; ask again once "
                                     "your profile has a bit more detail and I'll tailor it further.",
                "total_estimated_weeks": 12,
                "phases": [{
                    "title": "Get Oriented", "duration_weeks": 4,
                    "goal": "Understand the sustainability sector and identify your target niche.",
                    "milestones": [
                        {"title": "Complete your profile", "description": "Tell me your target role, location, and current background so I can tailor this properly.", "skill_related": "", "effort": "low"}
                    ]
                }]
            }

    # 2. Attach real, verified resources to each milestone (skill-matched where possible).
    gap_resources = {g.get("skill",""): g.get("resources", []) for g in gaps}
    for phase in roadmap.get("phases", []):
        for m in phase.get("milestones", []):
            skill = m.get("skill_related", "")
            if skill and gap_resources.get(skill):
                m["resources"] = gap_resources[skill][:2]
            else:
                found = rich_local_search(m.get("title", target or "sustainability"), location, "courses",
                                           target_role=target, sector=current_role)
                real = [f for f in found if f.get("type") != "search_fallback"][:1]
                m["resources"] = [{"title": r.get("title",""), "url": r.get("url","")} for r in real]

    # 3. Persist the structured plan (for progress display) and seed trackable milestones.
    roadmap["generated_at"] = datetime.now().isoformat()
    roadmap["target_role"]  = target
    roadmap["current_role"] = current_role
    vault["roadmap"] = roadmap
    _seed_roadmap_milestones(uid, vault, roadmap)
    save_vault(uid, vault)

    # 4. Render the HTML report and file it under Reports as before.
    topic  = target or re.sub(r'\b(roadmap|for|generate|create|make|a|the|map|my|career|transition)\b',
                               '', user_input, flags=re.IGNORECASE).strip() or "sustainability"
    folder = re.sub(r"[^a-z0-9]", "-", topic.lower())[:40].strip("-") or f"roadmap-{int(time.time())}"
    ws = os.path.join(SKILLS_DIR, folder)
    os.makedirs(ws, exist_ok=True)
    page = _render_roadmap_html(roadmap, target or topic)
    with open(os.path.join(ws, "report.html"), "w", encoding="utf-8") as f:
        f.write(page)
    path = f"/view-skill/{folder}/report"
    add_report(uid, f"Roadmap: {(target or topic).title()}", path, "roadmap")

    award_xp(uid, "roadmap_generated", vault)
    _grant_badge(uid, "road_builder", vault)
    save_vault(uid, vault)

    store_vector(uid,
        f"Generated phased roadmap for: {target or topic}. "
        f"{len(roadmap.get('phases',[]))} phases, ~{roadmap.get('total_estimated_weeks','?')} weeks.",
        "roadmap", importance=8)

    return {
        "path": path,
        "narrative_summary": roadmap.get("narrative_summary", ""),
        "total_estimated_weeks": roadmap.get("total_estimated_weeks", ""),
        "phase_titles": [p.get("title", "") for p in roadmap.get("phases", [])],
        "milestone_count": sum(len(p.get("milestones", [])) for p in roadmap.get("phases", [])),
    }

import re as _loc_re

def extract_location(text: str, vault_location: str = "") -> str:
    """Extract postcode, area, or use saved location."""
    if vault_location: return vault_location
    # Full UK postcode
    m = _loc_re.search(r"[A-Z]{1,2}[0-9][0-9A-Z]?\s*[0-9][A-Z]{2}", text, _loc_re.I)
    if m: return m.group(0).upper().strip()
    # Partial postcode
    m = _loc_re.search(r"\b([A-Z]{1,2}[0-9]{1,2}[A-Z]?)\b", text, _loc_re.I)
    if m: return m.group(1).upper()
    # "near X", "in X"
    m = _loc_re.search(r"(?:near|in|around|close to)\s+([A-Za-z][a-zA-Z\s]{2,20})", text)
    if m: return m.group(1).strip()
    return "London"

def rich_local_search(query: str, location: str, stype: str = "volunteering",
                       target_role: str = "", sector: str = "",
                       limitations: list = None, work_type: str = "") -> list:
    """Deep search for local resources — UK-defaulted, relevance- and
    freshness-filtered, and link-verified before being returned (Fix #1, #4, #6)."""
    limitations = limitations or []
    loc_ctx = f"{location}, UK" if location and "uk" not in location.lower() and "london" not in location.lower() else location
    BLOCKED = {"facebook.com","twitter.com","instagram.com","youtube.com","wikipedia.org"}
    constraint = ""
    if work_type: constraint += f" {work_type}"
    if limitations: constraint += " " + " ".join(limitations[:2])

    # `query` is often just "{target_role} {sector}" (same info as the named
    # params below) — but callers like the skill-gap resource fetcher pass a
    # SPECIFIC skill name here (e.g. "Data Analysis (Environmental Focus)") that
    # is otherwise completely ignored by both search and relevance filtering
    # below, which is why gap-specific resource lookups silently degraded into
    # generic "{target_role} course" results. Detect that case and anchor on it.
    _generic_query_forms = {
        (target_role or "").strip().lower(),
        (sector or "").strip().lower(),
        f"{target_role} {sector}".strip().lower(),
        f"{sector} {target_role}".strip().lower(),
    }
    is_specific_query = bool((query or "").strip()) and query.strip().lower() not in _generic_query_forms

    queries = {
        "volunteering": [
            f"volunteering {target_role or 'sustainability'} {loc_ctx} apply site:do-it.org OR site:ncvo.org.uk OR site:volunteering-matters.org.uk",
            f"environmental volunteering {loc_ctx} opportunities apply {sector}{constraint}",
            f"conservation volunteer {loc_ctx} join contact",
            f"green volunteering {location} how to get involved site:.org.uk OR site:.co.uk",
        ],
        "jobs": [
            f"{target_role or 'sustainability'} jobs {loc_ctx} apply site:reed.co.uk OR site:greenjobs.co.uk{constraint}",
            f"{location} {target_role or 'sustainability'} vacancies 2026 site:environmentjob.co.uk",
            # NOTE: deliberately anchored on target_role, never a bare "{sector} jobs"
            # search — sector is the user's OLD/current background (e.g. "tech"), and a
            # bare search on it just returns jobs in their old field, not their target.
            f"{target_role or 'sustainability'} roles valuing {sector or 'transferable'} background {loc_ctx} 2026 apply{constraint}",
        ],
        "courses": [
            f"free {target_role or 'sustainability'} course online UK 2026 site:futurelearn.com OR site:coursera.org OR site:openlearn.open.ac.uk",
            f"free green skills training {target_role or sector} {loc_ctx} 2026",
            f"{target_role or 'sustainability'} CPD free {location} online certificate 2026",
        ],
        "events": [
            f"{target_role or 'sustainability'} networking event {loc_ctx} 2026 site:eventbrite.co.uk",
            f"green careers event {sector or location} 2026",
            f"{sector or 'climate action'} meetup {loc_ctx} 2026 eventbrite OR meetup",
        ],
        "webinars": [
            f"{target_role or 'sustainability'} webinar 2026 register free online",
            f"{sector or 'green economy'} webinar UK 2026 sign up",
            f"{target_role or 'sustainability'} online talk panel 2026 register",
        ],
        "community": [
            f"{target_role or 'sustainability'} community group {loc_ctx} join",
            f"environmental action group {location} {sector} how to join",
            f"Friends of the Earth local group {location}",
        ],
    }

    query_list = list(queries.get(stype, queries["volunteering"]))
    if is_specific_query:
        specific_query_map = {
            "courses":      f"free {query} course training UK 2026",
            "volunteering": f"{query} volunteering {loc_ctx} apply",
            "jobs":         f"{query} jobs {loc_ctx} apply 2026",
            "events":       f"{query} event {loc_ctx} 2026",
            "webinars":     f"{query} webinar 2026 register",
            "community":    f"{query} community group {loc_ctx} join",
        }
        query_list = [specific_query_map.get(stype, f"{query} {stype} {loc_ctx}")] + query_list

    # When we have a specific topic (e.g. a skill gap), relevance must be judged
    # against THAT topic, not just the broad target_role/sector — a course that's
    # generically "sustainability" but has nothing to do with the actual skill
    # gap should not pass.
    relevance_focus = query if is_specific_query else " ".join([target_role or "", sector or ""]).strip()
    relevance_sector = "" if is_specific_query else sector

    results = []
    seen    = set()
    for q in query_list:
        batch = ddgs_search(q, n=6)
        if relevance_focus:
            _prewarm_relevance(batch, relevance_focus, relevance_sector)
        for r in batch:
            url = r.get("url","")
            if not url or url in seen: continue
            domain = url.split("/")[2] if "/" in url else ""
            if any(b in domain for b in BLOCKED): continue
            if relevance_focus:
                if not _is_relevant(r, relevance_focus, relevance_sector):
                    continue
            if not _is_fresh(r):
                continue
            path_depth = url.count("/") - 2
            seen.add(url)
            results.append({
                "url":        url,
                "title":      r.get("title",""),
                "body":       r.get("body","")[:500],
                "type":       stype,
                "path_depth": path_depth,
            })
        if len(results) >= 10: break

    results.sort(key=lambda x: x.get("path_depth",0), reverse=True)
    fallback_focus = query if is_specific_query else (target_role or sector or "sustainability")
    # Never publish an unverified/dead link — check liveness before returning (Fix #1).
    return verify_resources(results, f"{stype} {fallback_focus} {location}", keep=8)


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR  — SSE streaming with job cards
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE SEARCHER — finds real, live, tailored resources (Fix #1, #2, #4, #6)
# ─────────────────────────────────────────────────────────────────────────────
_CONFIRMATION_WORDS = {
    "yes","yeah","yep","yup","sure","ok","okay","please","go ahead","sounds good",
    "let's do it","lets do it","do it","yes please","sounds great","perfect","great idea"
}

def _looks_like_confirmation(text: str) -> bool:
    """A short affirmative reply (<=5 words) that only makes sense as an answer
    to whatever the assistant just offered, e.g. 'yes' / 'sure' / 'ok'."""
    t = text.strip().lower().rstrip("!.")
    if not t:
        return False
    if t in _CONFIRMATION_WORDS:
        return True
    return len(t.split()) <= 5 and any(t.startswith(w) for w in _CONFIRMATION_WORDS)

TRANSFERABLE_SKILLS_PROMPT = """You are a sustainability career transition expert. This person's
experience is NOT in sustainability yet, but many skills genuinely transfer. Given the market
reality that most green jobs need adjacent skills (data, stakeholder management, communications,
project delivery, regulatory literacy, community organising) far more than sector-specific ones,
identify what from this person's actual background genuinely applies — do NOT force a connection
that isn't real; leave it out if there's no genuine link.

Background / CV summary: {background}
Previous sector: {sector}
Hobbies/interests: {hobbies}
Target role: {target_role}

Return ONLY valid JSON:
{{"mappings": [
  {{"source": "specific thing from their background, e.g. 'Ran product launches at a tech startup'",
    "skill": "the transferable skill, e.g. 'Cross-functional stakeholder management'",
    "relevance": "one concrete sentence on why this matters for {target_role}"}}
]}}"""

def map_transferable_skills(uid: str) -> list:
    """Map background/CV experience (not just hobbies) onto genuine
    sustainability-relevant transferable skills. Stored separately from
    hobby_skill_map so existing prompt code that reads that field is untouched.
    Also folds each mapped skill into the main vault['skills'] list with a
    recorded source (see _add_skill/_attribute_skill), so a skill inferred
    from a hobby or a chat-described experience shows up — with its origin —
    on the profile card the same way a CV-derived skill would."""
    vault      = load_vault(uid)
    exp_lines  = [f"{e.get('title','')} at {e.get('company','')}: {e.get('description','')}".strip(": ").strip()
                  for e in vault.get("experience", []) if e.get("title") or e.get("description")]
    background = " | ".join(p for p in [vault.get("cv_summary", "")] + exp_lines if p)
    sector     = vault.get("sector", "")
    hobbies    = ", ".join(_safe_str_list(vault.get("hobbies"))) or "none stated"
    target     = vault.get("target_role", "") or "a sustainability role"
    if not background and not sector and hobbies == "none stated":
        return []

    prompt = TRANSFERABLE_SKILLS_PROMPT.format(
        background=background or "not provided", sector=sector or "not stated",
        hobbies=hobbies, target_role=target
    )
    raw = query_ai(prompt)
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        mappings = data.get("mappings", [])
    except Exception:
        mappings = []

    if mappings:
        vault["transferable_skills"] = mappings
        for m in mappings:
            skill = (m.get("skill") or "").strip()
            if not skill:
                continue
            detail = " — ".join(p for p in [m.get("source", ""), m.get("relevance", "")] if p)
            _add_skill(vault, skill, "inferred", detail)
        save_vault(uid, vault)
        store_vector(uid, f"Transferable skills mapped: {json.dumps(mappings)[:400]}",
                     "transferable_skills", importance=8)
    return mappings

def _is_relevant(r, target_role: str, sector: str) -> bool:
    """Filter out noise (e.g. '3D modelling', TV series, unrelated hobby content)
    by requiring the result to actually relate to the target role/sector.
    Uses semantic similarity (all-MiniLM-L6-v2) rather than raw keyword overlap,
    so e.g. 'net zero programme officer' still passes for a target role of
    'sustainability consultant' even without a shared exact word, while a
    literal keyword hit that's actually off-topic (e.g. 'green' as in golf)
    isn't enough on its own — deliberately no keyword fast-path here, since
    single-word splits of target_role/sector are too generic to trust alone.
    Logs the score for later threshold calibration — see
    logs/match_calibration.jsonl."""
    text  = (r.get("title","") + " " + r.get("body","")).strip()
    query = " ".join([target_role or "", sector or ""]).strip()
    if not query:
        return True
    if not text:
        return False
    score    = _cosine_sim(query, text[:600])
    # NOTE: all-MiniLM-L6-v2 cosine similarity between genuinely unrelated short
    # texts commonly sits in the 0.15-0.35 range due to general English overlap,
    # so 0.30 was letting through results with no real topical connection (e.g.
    # immigration pages for a "data analysis" course search). Raised pending
    # further calibration from logs/match_calibration.jsonl.
    decision = score >= 0.40
    _log_calibration("relevance_signal", {"query": query, "sim_score": round(score, 3),
                                           "decision": decision})
    return decision

def _prewarm_relevance(results, target_role: str, sector: str) -> None:
    """Batch-embeds the query plus every candidate result's text in ONE
    embed_fn call, before the per-result _is_relevant() loop runs — so
    filtering a page of 15-25 search results costs one embedding call
    instead of one per result."""
    query = " ".join([target_role or "", sector or ""]).strip()
    if not query:
        return
    texts = [(r.get("title", "") + " " + r.get("body", "")).strip()[:600] for r in results]
    _embed_texts([query] + [t for t in texts if t])

def find_tailored_resources(target_role: str, sector: str, location: str = "London",
                             limitations: list = None, work_type: str = "") -> list:
    """Search for REAL, live, current (2025/2026) resources specific to the
    user's target role, across ALL resource types — courses, volunteering,
    jobs, events, webinars, and community — filtered by their stated
    constraints, and link-verified (Fix #1, #4, #6)."""
    resources = []
    seen = set()
    for stype in ["courses", "volunteering", "jobs", "events", "webinars", "community"]:
        batch = rich_local_search(f"{target_role} {sector}", location, stype,
                                   target_role=target_role, sector=sector,
                                   limitations=limitations, work_type=work_type)
        for r in batch:
            url = r.get("url","")
            if not url or url in seen:
                continue
            seen.add(url)
            resources.append({
                "url":      url,
                "title":    r.get("title",""),
                "body":     r.get("body","")[:200],
                "type":     stype,
                "verified": r.get("verified", True),
            })
    return resources[:24]

def find_specific_jobs(target_role: str, sector: str, skills: list, location: str = "London",
                        salary_range: str = "", work_type: str = "", limitations: list = None) -> list:
    """Search for ACTUAL job postings — not general websites — filtered for
    relevance, freshness, the user's salary/work-type preferences, and
    verified as live before being returned (Fix #1, #4, #6)."""
    limitations = limitations or []
    all_jobs = []
    skill_str = skills[0] if skills else ""
    constraint = f" {work_type}" if work_type else ""
    if salary_range: constraint += f" {salary_range}"

    searches = [
        f'{target_role} {location} job apply 2026 site:linkedin.com/jobs{constraint}',
        f'{target_role} {location} apply site:uk.indeed.com{constraint}',
        f'{target_role} {location} site:reed.co.uk{constraint}',
        f'{target_role} {sector} {location} vacancy 2026{constraint}',
        f'{target_role} {sector} {location} hiring 2026',
        f'{target_role} {location} site:jobs.environmentjob.co.uk',
        f'{target_role} {location} site:greenjobs.co.uk',
        f'{target_role} {location} site:charityjob.co.uk',
    ]
    if skill_str:
        searches.insert(0, f'{target_role} {skill_str} {location} job apply now')

    seen = set()
    for q in searches:
        results = ddgs_search(q, n=5)
        _prewarm_relevance(results, target_role, sector)
        for r in results:
            url = r.get("url","")
            if not url or url in seen:
                continue
            if not _is_relevant(r, target_role, sector):
                continue
            if not _is_fresh(r):
                continue
            is_job_page = any(x in url.lower() for x in [
                "linkedin.com/jobs","indeed.co.uk/jobs","indeed.com/jobs",
                "reed.co.uk","jobs.","careers.","totaljobs","cv-library",
                "glassdoor","charityjob","sustainablebusiness","edie.net",
                "greenjobs","environmentjob","jobsgreenuk"
            ])
            if is_job_page or len(all_jobs) < 3:
                seen.add(url)
                all_jobs.append(r)
        if len(all_jobs) >= 8:
            break
    # Never publish a dead/unverified job link (Fix #1).
    return verify_resources(all_jobs, f"{target_role} jobs {location}", keep=6)

# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — Comprehensive advisor with persona system
# ─────────────────────────────────────────────────────────────────────────────
class Orchestrator:

    def _build_context(self, uid: str, query: str = "") -> dict:
        """Always read full memory before acting. Fix #6."""
        vault    = load_vault(uid)
        recall   = recall_vector(uid, query or "career background goals", n=6)
        learning = vault.get("learning", {})
        journey  = vault.get("journey", {})
        msgs     = vault.get("messages", [])

        return {
            "vault":         vault,
            "name":          vault.get("name","") or (vault.get("email","").split("@")[0].title() if vault.get("email") else ""),
            "email":         vault.get("email",""),
            "current_bg":    vault.get("sector",""),         # what they DO now
            "target_role":   vault.get("target_role",""),    # what they WANT
            "cv_summary":    vault.get("cv_summary",""),
            "goals":         vault.get("goals",[]),
            "skills":        vault.get("skills",[]),
            "emotion":       learning.get("current_emotion","neutral"),
            "tone":          learning.get("preferred_tone","warm"),
            "blockers":      learning.get("blockers",[]),
            "insights":      learning.get("key_insights",[]),
            "exchanges":     learning.get("total_exchanges",0),
            "progress":      journey.get("progress_pct",0),
            "assessed":      journey.get("assessed",False),
            "skills_gaps":   journey.get("skills_gaps",[]),
            "milestones":    journey.get("milestones",[]),
            "tracked_jobs":  vault.get("tracked_jobs",[]),
            "recall":        recall,
            "xp":            journey.get("xp",0),
            "recent_msgs":   msgs[-12:],
            "is_returning":  len(msgs) > 0,
            "location":      vault.get("location","London"),
            "aspirations":   vault.get("aspirations",[]),
            "hobbies":       vault.get("hobbies",[]),
            "dream_job":     vault.get("dream_job",""),
            "values":        vault.get("values",[]),
            "culture_preference": vault.get("culture_preference",""),
            "salary_range":  vault.get("salary_range",""),
            "work_type":     vault.get("work_type",""),
            "hobby_skill_map": vault.get("hobby_skill_map",[]),
            "limitations":   vault.get("limitations",[]),
        }

    def _classify_intent(self, text: str, ctx: dict) -> str:
        """
        Classify what the user actually wants. Returns one of:
        'conversational' | 'job_search' | 'roadmap' | 'resources' |
        'coaching' | 'cv_help' | 'preferences' | 'general_question'
        """
        if _looks_like_confirmation(text):
            last_assistant_msg = ""
            for m in reversed(ctx.get("recent_msgs", [])):
                if m.get("role") == "assistant":
                    last_assistant_msg = m.get("content", "")
                    break
            if last_assistant_msg:
                return self._classify_intent(last_assistant_msg, ctx)

        lower = text.lower().strip()

        # Explicit job search (must be very specific)
        if any(k in lower for k in ["find me jobs","find me a job","find me roles",
            "search for jobs","show me jobs","job listings","who is hiring",
            "look for jobs","find roles","job search","show me vacancies",
            "what jobs can i","what jobs could i","which jobs can i","which jobs suit",
            "what jobs fit","what jobs would suit","what jobs match","jobs can i go for",
            "jobs suit me","jobs fit me","what roles can i","which roles can i",
            "what can i apply for","what should i apply for"]):
            return "job_search"

        # Explicit roadmap
        if any(k in lower for k in ["roadmap","30-day","training plan","study plan",
            "career plan","curriculum","learning plan","action plan"]):
            return "roadmap"

        # Transferable skills — "does my background count" type questions
        if any(k in lower for k in ["transferable skill","transfer to sustainability","does my experience",
            "does my background","what skills carry over","what carries over","what applies from my",
            "skills from my background","skills count"]):
            return "transferable_skills"

        # CV help
        if any(k in lower for k in ["cv help","review my cv","improve my cv","cv advice",
            "cv feedback","my resume","update my cv","cv for","rewrite my cv"]):
            return "cv_help"

        # Local / postcode / area search
        import re as _r2
        has_postcode = bool(_r2.search(r"[A-Z]{1,2}[0-9][0-9A-Z]?\s*[0-9]?[A-Z]{0,2}", lower, _r2.I))
        local_triggers = ["near me","near ","in my area","close to","local","around me","postcode","nw","sw","se","ec","wc","e1","n1","w1"]
        vol_triggers   = ["volunteer","sustainability","environmental","green","conservation","community","course","event","group","network"]
        if (has_postcode or any(k in lower for k in local_triggers)) and any(k in lower for k in vol_triggers):
            return "local_search"

        # Resources / general
        if any(k in lower for k in ["free course","volunteering","volunteer","resources",
            "community group","networking","training","upskill","certif","fellowship",
            "bootcamp","workshop","where can i","how can i get involved","opportunities",
            "how do i find","local group","what courses"]):
            return "resources"

        # Preferences update
        if any(k in lower for k in ["update my preferences","my goal is","i want to become",
            "i am transitioning","i'm transitioning","my target is","change my target",
            "my sector is","my background is","i now work"]):
            return "preferences"

        # Coaching
        if any(k in lower for k in ["coach me","coaching","interview prep","mock interview",
            "salary negotiation","help me prepare for","practice interview"]):
            return "coaching"

        # Everything else — conversational (Fix #2: default to chat, not job search)
        return "conversational"

    def _select_persona(self, text: str, ctx: dict) -> dict:
        """
        Select the right persona and strategy based on what we know.
        Returns a dict with persona_name, tone, strategy, opening_guidance.
        """
        effective_text = text
        if _looks_like_confirmation(text):
            for m in reversed(ctx.get("recent_msgs", [])):
                if m.get("role") == "assistant":
                    effective_text = m.get("content", "") or text
                    break

        lower    = effective_text.lower()
        emotion  = ctx["emotion"]
        exchanges = ctx["exchanges"]
        name     = ctx["name"]

        # Emotional continuity: if the user was distressed very recently and this
        # message doesn't clearly signal recovery, carry a touch of that acknowledgment
        # forward even into a task-oriented persona, instead of snapping straight to "advisor".
        recent_emotions = [e.get("emotion") for e in ctx["vault"].get("learning", {}).get("emotion_history", [])[-3:]]
        recently_distressed = any(e in ["frustrated", "anxious", "stuck"] for e in recent_emotions[:-1])

        # Emotional distress — psychologist first
        if emotion in ["frustrated","anxious","stuck"] or any(k in lower for k in [
            "anxious","worried","scared","nervous","stressed","overwhelmed",
            "imposter","doubt","not confident","don't know","don't know where",
            "lost","confused","hopeless","giving up","can't do this"]):
            return {
                "name": "psychologist",
                "style": "warm and deeply empathetic",
                "strategy": (
                    "FIRST: Acknowledge their feelings genuinely — not with platitudes. "
                    "Show you understand what they're going through specifically. "
                    "DO NOT jump to advice. Ask one open question to understand more. "
                    "Only if they seem ready: gently offer one small step. "
                    "End by letting them know you're here, however they need you."
                ),
                "should_suggest_jobs": False,
                "should_add_resources": False,
            }

        # Celebrating / excited — friend
        if emotion in ["excited","grateful"] or any(k in lower for k in [
            "got the job","got an interview","offer","accepted","they said yes",
            "i did it","i passed","great news","exciting","so happy","thrilled"]):
            return {
                "name": "friend",
                "style": "genuinely excited and celebratory",
                "strategy": (
                    "FIRST: Celebrate with them genuinely. Be specific about what's great. "
                    "Match their energy. Then help them capitalise on the momentum. "
                    "Ask what they need next. Keep it light and energising."
                ),
                "should_suggest_jobs": False,
                "should_add_resources": True,
            }

        # Rejection / failure — compassionate coach
        if any(k in lower for k in ["rejected","didn't get","not selected","failed",
            "turned down","no offer","unsuccessful"]):
            return {
                "name": "coach",
                "style": "compassionate but forward-focused",
                "strategy": (
                    "FIRST: Hold space. Don't rush past the disappointment. "
                    "Name what happened clearly and with empathy. "
                    "Then: what can be learned? What did they do well? "
                    "End with one specific, achievable next step. "
                    "Reference their specific role and background."
                ),
                "should_suggest_jobs": True,
                "should_add_resources": False,
            }

        # New user — curious friend getting to know them
        if exchanges < 4 or not ctx["current_bg"]:
            # Determine what we still need to know
            missing = []
            if not ctx.get("current_bg"):    missing.append("what they currently do")
            if not ctx.get("target_role"):   missing.append("their target role")
            if not ctx.get("location"):      missing.append("their location")
            if not ctx.get("goals"):         missing.append("their career goals")
            if not ctx.get("dream_job"):      missing.append("their dream job")
            if not ctx.get("salary_range"):   missing.append("their salary expectations")
            if not ctx.get("work_type"):      missing.append("whether they need remote/hybrid/onsite")
            if not ctx["vault"].get("limitations"): missing.append("any constraints — location, hours, caring responsibilities")

            focus = f"Your TOP priority: find out {missing[0]}." if missing else "You know them well — focus on moving them forward."
            return {
                "name": "curious_friend",
                "style": "warm, curious, and genuinely interested in them",
                "strategy": (
                    f"{focus} "
                    "Ask ONE specific open question to learn about them — not multiple at once. "
                    "When they share info, affirm it warmly and tell them you've noted it. "
                    "Make them feel heard. Build trust before offering advice. "
                    "DO NOT give career advice until you understand their background."
                ),
                "should_suggest_jobs": False,
                "should_add_resources": False,
            }

        # General question / factual — knowledgeable advisor
        if effective_text.strip().endswith("?") or any(k in lower for k in [
            "what is","what are","how do","how does","tell me about","explain",
            "what's the difference","is it worth","can you tell me"]):
            continuity = (" Briefly acknowledge how they were feeling a moment ago before answering, "
                          "in one short clause — don't dwell on it.") if recently_distressed else ""
            return {
                "name": "advisor",
                "style": "knowledgeable, clear, and personal",
                "strategy": (
                    "Answer their question directly and thoroughly. "
                    "Connect the answer to their specific situation where relevant. "
                    "Use examples. Be concrete. Then offer to go deeper or help them act on it."
                    + continuity
                ),
                "should_suggest_jobs": False,
                "should_add_resources": True,
            }

        # Career strategy — work coach
        continuity2 = (" They seemed to be struggling recently — weave in a brief note of encouragement, "
                       "not a big speech, just acknowledgment that you remember.") if recently_distressed else ""
        return {
            "name": "coach",
            "style": "direct, strategic, and personally invested",
            "strategy": (
                "You know this person. Reference their specific background, goals, and progress. "
                "Give structured, actionable guidance. "
                "Connect everything back to their green career goal. "
                "Be honest — if they need to hear something hard, say it with kindness."
                + continuity2
            ),
            "should_suggest_jobs": ctx.get("target_role","") != "",
            "should_add_resources": True,
        }

    def _build_prompt(self, user_input: str, ctx: dict, persona: dict,
                      resources: list = None, intent: str = "conversational") -> str:
        """Build the full LLM prompt with all context injected."""
        name   = ctx["name"]
        bg     = ctx["current_bg"]
        target = ctx["target_role"]

        # Profile block — always explicit about transition status (Fix #6)
        profile = []
        if name:     profile.append(f"Name: {name}")
        if bg:       profile.append(f"Current background: {bg} (this is where they ARE, not where they want to go)")
        if target:   profile.append(f"Career goal / target role: {target} (this is where they WANT to get to)")
        if bg and target and bg != target:
            profile.append(f"STATUS: They are TRANSITIONING from {bg} INTO {target}. They do NOT yet work in {target}.")
        if ctx["cv_summary"]:  profile.append(f"CV summary: {ctx['cv_summary'][:200]}")
        if ctx["skills"]:      profile.append(f"Known skills: {', '.join(_safe_str_list(ctx['skills'])[:5])}")
        if ctx["goals"]:       profile.append(f"Goals: {', '.join(_safe_str_list(ctx['goals'])[:3])}")
        if ctx["blockers"]:    profile.append(f"Known blockers/fears: {', '.join(_safe_str_list(ctx['blockers'])[:3])}")
        if ctx["insights"]:    profile.append(f"What you've learned about them: {'. '.join(ctx['insights'][-3:])}")
        if ctx.get("progress",0) > 0 and intent in ("roadmap","coaching","job_search"):
            profile.append(f"Journey progress: {ctx['progress']}% toward target")
        if ctx.get("dream_job"):       profile.append(f"Dream job: {ctx['dream_job']}")
        if ctx.get("hobbies"):         profile.append(f"Hobbies (potential skills): {', '.join(_safe_str_list(ctx['hobbies'])[:4])}")
        if ctx.get("values"):          profile.append(f"Work values: {', '.join(_safe_str_list(ctx['values'])[:3])}")
        if ctx.get("culture_preference"): profile.append(f"Company culture preference: {ctx['culture_preference']}")
        if ctx.get("salary_range"):    profile.append(f"Expected salary: {ctx['salary_range']}")
        if ctx.get("work_type"):       profile.append(f"Work preference: {ctx['work_type']}")
        if ctx.get("location"):        profile.append(f"Location: {ctx['location']}")
        if ctx.get("hobby_skill_map"):
            hmap = "; ".join(f"{h['hobby']}: {', '.join(_safe_str_list(h.get('skills'))[:2])}" for h in ctx["hobby_skill_map"][:2] if h.get("hobby"))
            if hmap: profile.append(f"Skills from hobbies: {hmap}")
        if ctx.get("location"):        profile.append(f"User location: {ctx['location']} — all local recommendations must be near here")
        if ctx.get("recall"):          profile.append(f"Relevant memory: {ctx['recall'][:300]}")
        profile_block = "\n".join(profile) if profile else "Profile not yet complete — learn about them."

        # Conversation history
        history_lines = []
        for m in ctx["recent_msgs"][-10:]:
            r = m.get("role","")
            c = m.get("content","")[:300]
            if r == "user":       history_lines.append(f"User: {c}")
            elif r == "assistant": history_lines.append(f"Garden AI: {c}")

        # Resource block if available
        resource_lines = []
        if resources:
            resource_lines.append("REAL RESOURCES/LINKS FOUND (use these, not invented ones):")
            for r in resources[:5]:
                t = r.get("title","")
                u = r.get("url","")
                b = r.get("body","")[:100]
                resource_lines.append(f"- [{t}]({u}): {b}")
        resource_block = "\n".join(resource_lines)
        history_str = chr(10).join(history_lines) if history_lines else "This is the start of the conversation."
        resource_str = resource_block

        last_assistant_msg = ""
        for m in reversed(ctx["recent_msgs"]):
            if m.get("role") == "assistant":
                last_assistant_msg = m.get("content","")[:500]
                break

        confirmation_note = ""
        if last_assistant_msg and _looks_like_confirmation(user_input):
            confirmation_note = (
                f"\nIMPORTANT: The user's message ('{user_input}') is a short confirmation "
                f"(yes/ok/sure/etc). It refers to what YOU offered in your last message below — "
                f"do NOT ask the same question again or restart the conversation. "
                f"Directly follow through on that specific offer with concrete detail.\n"
                f"YOUR LAST MESSAGE WAS: \"{last_assistant_msg}\"\n"
            )

        return f"""You are Garden AI — a personalised sustainability career companion built for the London green economy.
Today you are acting as a {persona['name']} with a {persona['style']} approach.

STRATEGY FOR THIS RESPONSE:
{persona['strategy']}

USER PROFILE (read carefully before responding):
{profile_block}

CONVERSATION HISTORY:
{history_str}
{resource_block}
{confirmation_note}
User just said: "{user_input}"

RULES — read carefully before every response:
1. LISTEN FIRST. Read the user's message carefully. If it's emotional, acknowledge before advising. If it's a question, answer it directly first.
2. NEVER be generic. Every sentence must reference their specific background, target role, location, or something personal you know about them.
3. NEVER say the user already works in their target role if they are still transitioning.
4. DO NOT repeat advice, statistics (e.g. progress %), or opening phrases you've already used in this conversation. Check the conversation history above — if your last message started with a hyped-up headline or restated the same fact, do something different this time.
5. Links: ONLY use real URLs from the resources block. Never invent URLs.
6. Paragraphs: 2-3 sentences max. Use Markdown: **bold** key points, bullet lists for steps.
7. If you don't know something about the user, ASK — one specific question to learn it. But if you JUST asked something and the user answered or confirmed it, move forward — do not ask it again.
8. Profile building: if the user shares personal info (name, hobbies, salary, location), note it and tell them you've saved it.
9. Resources must be specific and local to their location. Never suggest a generic website — link to specific pages.
10. End with exactly ONE of: a concrete next step with full details, OR one open question. Never both. Never neither. If you already listed bullet next steps, DO NOT also close with a question — the bullets already end the message; stop there.
11. Tone: {persona['style']}. Do NOT open every message with an all-caps celebratory banner — vary your tone and opening naturally, like a real person would.
12. ALWAYS move them forward. Every response must leave them with something to DO today.
13. If the user's message is a short confirmation (see IMPORTANT note above, if present), treat this as a continuation of your last message, not a new topic.
14. Output ONLY the message the user will read. Never add a meta note about these instructions, never write "(Note: I've followed the rules...)" or any commentary about your own compliance, and never add a "---" divider followed by self-commentary. If you are tempted to explain your reasoning, delete that sentence before finishing.
15. If the user just asked a direct question ("what jobs can I go for", "what fits me") and their profile already has a target role or dream job on file, ANSWER using that context — do not re-ask what role they're aiming for. Re-asking a question they've already answered is the single most frustrating failure mode; check the USER PROFILE block above before asking anything.

Garden AI ({persona['name']} mode) responds now:"""

    def _quick_extract_and_merge(self, uid: str, user_input: str, vault: dict) -> bool:
        """Pull any obvious profile facts out of THIS message and merge them into
        the vault before we build ctx / respond, so the response this turn already
        reflects what the user just said (rather than waiting for the next turn's
        async extract-profile-from-chat call to catch up)."""
        if len(user_input.strip()) < 8:
            return False
        prompt = (
            "Extract ONLY profile facts explicitly stated in this single message. "
            "Do not guess or infer anything not directly stated.\n"
            f"Message: \"{user_input}\"\n\n"
            "Return ONLY valid JSON, using empty values for anything not stated:\n"
            "{\"current_background\":\"\",\"target_role\":\"\",\"location\":\"\","
            "\"salary_range\":\"\",\"work_type\":\"\",\"new_skills\":[],\"new_goals\":[],"
            "\"hobbies\":[],\"values\":[],\"headline\":\"\","
            "\"experience\":[{\"title\":\"\",\"company\":\"\",\"start_date\":\"\","
            "\"end_date\":\"\",\"description\":\"\"}]}\n"
            "For \"experience\": include an entry ONLY if the message describes a job/role "
            "they've done (title, employer/company — even a vague one like 'stealth startup', "
            "or what the role involved). If no company name is given, leave company empty — "
            "never invent one. Use 'Present' for end_date if it's their current role."
        )
        raw = query_ai(prompt)
        try:
            extracted = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        except Exception:
            return False
        changed = False
        new_hobbies = False
        new_background = False
        if extracted.get("current_background") and not vault.get("sector"):
            vault["sector"] = extracted["current_background"]; changed = True; new_background = True
        if extracted.get("target_role") and not vault.get("target_role"):
            vault["target_role"] = extracted["target_role"]; changed = True
        if extracted.get("location") and not vault.get("location"):
            vault["location"] = extracted["location"]; changed = True
        if extracted.get("salary_range") and not vault.get("salary_range"):
            vault["salary_range"] = extracted["salary_range"]; changed = True
        if extracted.get("work_type") and not vault.get("work_type"):
            vault["work_type"] = extracted["work_type"]; changed = True
        for sk in _safe_str_list(extracted.get("new_skills")):
            if _add_skill(vault, sk, "chat", user_input):
                changed = True
        for g in _safe_str_list(extracted.get("new_goals")):
            if g not in vault.get("goals", []):
                vault.setdefault("goals", []).append(g); changed = True
        for h in _safe_str_list(extracted.get("hobbies")):
            if h not in vault.get("hobbies", []):
                vault.setdefault("hobbies", []).append(h); changed = True; new_hobbies = True
        for val in _safe_str_list(extracted.get("values")):
            if val not in vault.get("values", []):
                vault.setdefault("values", []).append(val); changed = True
        if extracted.get("headline") and not vault.get("headline"):
            vault["headline"] = extracted["headline"]; changed = True
        new_experience  = _upsert_experience(vault, extracted.get("experience") or [])
        if new_experience:
            changed = True
        if changed:
            save_vault(uid, vault)
            store_vector(uid, f"Same-turn profile update: {json.dumps(extracted)}", "profile", 8)
            if new_hobbies:
                # A hobby just answered is only useful once it's mapped to a
                # skill — do it now, in this same turn, rather than leaving it
                # sitting unused until some later manual "re-map" action.
                try:
                    hmap = query_ai(
                        "Given these hobbies/interests, identify any GENUINE transferable "
                        f"skill toward a sustainability career: {', '.join(_safe_str_list(vault.get('hobbies')))}. "
                        "Return ONLY JSON: {\"mappings\":[{\"hobby\":\"\",\"skills\":[]}]} — "
                        "leave a hobby out entirely if there's no real link, don't force one."
                    )
                    data = json.loads(hmap[hmap.find("{"):hmap.rfind("}")+1])
                    if data.get("mappings"):
                        vault["hobby_skill_map"] = data["mappings"]
                        save_vault(uid, vault)
                except Exception:
                    pass
            # Anything that could plausibly reveal a genuine transferable
            # skill — a new hobby, a new/updated background, or a chat-
            # described experience — gets the SAME thorough, source-attributed
            # assessment used for CVs (map_transferable_skills), not just the
            # simpler hobby-only pass above. This is what lets "organised a
            # community fundraiser" mentioned in passing end up on the Skills
            # panel with its source, same as a CV-derived skill would.
            if new_hobbies or new_background or new_experience:
                try:
                    map_transferable_skills(uid)
                except Exception as e:
                    logging.warning(f"auto transferable-skills from chat: {e}")
        return changed

    def _check_stuck_resources(self, uid: str, user_input: str) -> str:
        """If the user seems to be struggling and has an in-progress resource,
        surface a small, relevant offer of help rather than generic advice."""
        lower = user_input.lower()
        struggle_signals = ["couldn't reach","can't reach","no reply","haven't heard",
                             "not sure how to","stuck on","struggling with","tried but",
                             "no response","ghosted","waiting to hear","don't know how to start"]
        if not any(s in lower for s in struggle_signals):
            return ""
        vault = load_vault(uid)
        in_progress = [r for r in vault.get("saved_resources", []) if r.get("status") == "in_progress"]
        if not in_progress:
            return ""
        r = in_progress[0]
        return (f"\nNOTE: The user has an in-progress resource — '{r.get('title','')}' ({r.get('type','')}). "
                f"If their struggle relates to this, offer concrete, specific next-step help "
                f"(e.g. a follow-up email template, alternative contacts, or how to escalate politely) "
                f"rather than generic encouragement.")

    def _profile_ready(self, ctx: dict) -> bool:
        """True once we actually know what this person is looking for.
        Without this, action intents fall back to a generic 'sustainability'
        search — which is exactly why resources/jobs used to feel generic
        and disconnected from the person."""
        return bool(ctx.get("target_role") or ctx.get("dream_job"))

    # ─────────────────────────────────────────────────────────────────────
    # DETERMINISTIC INTAKE SEQUENCE — mirrors the reference "Jack & Jill"
    # onboarding flow from the screenshots: one concrete question at a time,
    # in a fixed order, using CANNED text (never LLM-regenerated), so the
    # exact wording never drifts or repeats between turns. A step is skipped
    # the moment its field is known — from a direct answer, a CV upload, or
    # anything already on file — and is never re-asked afterward unless the
    # user brings it up again themselves (e.g. "actually, change my salary
    # target to ..." — that's a normal profile edit, handled elsewhere, not
    # a re-ask of this sequence).
    # ─────────────────────────────────────────────────────────────────────
    _INTAKE_SEQUENCE = [
        {
            "key": "target_role",
            "question": (
                "Before I go find jobs, courses, or resources for you, I want to make sure I get you the "
                "*right* ones — not generic ones. What kind of role are you looking for? Any specific job "
                "titles, sectors, or type of work on your radar — and what's the \"why\" behind the search?"
            ),
            "suggestions": ["I'm not sure yet — help me explore", "Renewable energy / EV roles", "Sustainability / ESG roles"],
        },
        {
            "key": "background",
            "question": (
                "No problem — often the best way to figure out the \"what\" is to look at the \"where from\". "
                "Do you have a CV you can drop in (using the '+' button), or would you rather just tell me "
                "what you've been doing for the last few years?"
            ),
            "suggestions": ["Upload my CV", "I'll describe it instead"],
            # Any of these counts as "background provided": a fact the user told
            # us directly (sector), something the AI extracted (cv_summary,
            # experience), OR simply the CV having been uploaded at all
            # (cv_uploaded / a "cv" document on file). That last pair matters
            # because cv_summary/sector/experience only get populated when the
            # AI extraction step succeeds — if the AI is down or returns
            # nothing usable, the CV was still received, and we must never
            # re-ask for it just because analysis failed.
            "satisfied": lambda v: bool(
                v.get("sector") or v.get("cv_summary") or v.get("experience")
                or v.get("cv_uploaded")
                or any(d.get("type") == "cv" for d in v.get("documents", []))
            ),
        },
        {
            "key": "location",
            "question": "Are you looking for roles in London specifically (hybrid/on-site), or open to remote/UK-wide?",
            "suggestions": ["London (hybrid/on-site)", "Remote, UK-wide", "Open to both"],
        },
        {
            "key": "salary_range",
            "question": "What's your ideal salary range? Knowing your floor and target helps me filter out roles that aren't the right fit.",
            "suggestions": ["Under £30k", "£30k–£50k", "£50k–£70k", "£70k+"],
        },
        {
            "key": "work_type",
            "question": "Do you need this to be remote, hybrid, or onsite?",
            "suggestions": ["Remote", "Hybrid", "Onsite"],
        },
        {
            "key": "culture_preference",
            "question": (
                "Last one for now: what kind of company culture are you after — an early-stage, "
                "all-hands-on-deck startup, or a more established company with more structure?"
            ),
            "suggestions": ["Early-stage startup", "Established company", "No strong preference"],
        },
    ]

    _ROLE_DISCOVERY_TRIGGERS = ["explore which", "explore what", "help me figure",
        "figure it out", "not sure yet", "not sure what role", "which roles fit",
        "which role fits", "help me explore", "don't know what role", "dont know what role",
        "not sure", "help me exp"]

    _DISTRESS_KEYWORDS = ["overwhelmed", "anxious", "stressed", "hopeless", "scared",
        "worried", "depressed", "struggling", "lost my job", "made redundant",
        "can't cope", "cant cope", "panicking", "burnt out", "burned out"]

    def _intake_satisfied(self, step: dict, vault: dict) -> bool:
        if step["key"] in vault.get("intake_skipped", []):
            return True
        custom = step.get("satisfied")
        if custom:
            return bool(custom(vault))
        return bool(vault.get(step["key"]))

    def _next_intake_step(self, vault: dict) -> dict:
        """First unsatisfied step in the sequence — or None if the whole
        intake brief is complete. A step only counts as done once a real
        value is in its vault field (or it's been explicitly skipped) —
        never merely because its question was displayed. This is what
        prevents a step whose answer failed to parse from being silently
        abandoned: it simply stays the 'next' step until it resolves."""
        for step in self._INTAKE_SEQUENCE:
            if not self._intake_satisfied(step, vault):
                return step
        return None

    def _intake_step_response(self, step: dict, vault: dict, uid: str) -> dict:
        """Return this step's canned question and record it as the pending
        step so the NEXT message is captured as its answer (see
        _capture_intake_answer). Deliberately does not mark the step
        resolved here — display alone is never enough."""
        vault["intake_pending"] = step["key"]
        save_vault(uid, vault)
        return {"response": step["question"], "suggestions": step["suggestions"], "intake_step": True}

    def _capture_intake_answer(self, pending_key: str, user_input: str, vault: dict, uid: str) -> None:
        """Resolve whichever intake step was left pending from the previous
        turn, BEFORE anything else runs this turn. This is a guaranteed
        fallback independent of the AI extractor: even if the generic
        free-text extractor fails to parse a short chip reply ('Remote'),
        a chip label ('Early-stage startup'), or the AI is briefly
        unavailable, a real value still lands in the field here — so the
        step is never left permanently empty just because it was 'asked'."""
        step = next((s for s in self._INTAKE_SEQUENCE if s["key"] == pending_key), None)
        if not step:
            return
        text = (user_input or "").strip()
        if not text:
            return
        lower = text.lower()
        # The "not sure yet, help me explore" pivot on target_role isn't an
        # answer to store — it's an explicit request to skip straight to
        # the background/CV question instead (mirrors the reference flow's
        # role -> CV/background pivot). Record it as skipped, not answered.
        if pending_key == "target_role" and any(t in lower for t in self._ROLE_DISCOVERY_TRIGGERS):
            if pending_key not in vault.get("intake_skipped", []):
                vault.setdefault("intake_skipped", []).append(pending_key)
            save_vault(uid, vault)
            return
        # Already resolved some other way while this was pending (e.g. a CV
        # upload landed mid-flight and satisfied "background") — nothing to
        # capture, don't overwrite it with the raw reply.
        if self._intake_satisfied(step, vault):
            return
        # Chip-driven steps: prefer the offered suggestion label if the reply
        # matches one (case-insensitive, either direction); otherwise fall
        # back to the raw text itself. Either way a concrete value lands in
        # the vault field — that's the guarantee the AI extractor doesn't have.
        matched = None
        for sug in step.get("suggestions", []):
            sl = sug.lower()
            if sl == lower or sl in lower or lower in sl:
                matched = sug
                break
        value = matched or text
        if step["key"] == "background":
            # "background" has no single scalar field of its own — it's
            # satisfied by sector/cv_summary/experience/cv_uploaded. If none
            # of those got populated (by the extractor or a CV upload),
            # store the raw description as sector so the step still resolves.
            if not vault.get("sector"):
                vault["sector"] = text
        else:
            vault[step["key"]] = value
        save_vault(uid, vault)

    def process(self, user_input: str, uid: str):

        """Generator yielding SSE lines."""

        yield _sse({"status": "🌱 Reading your profile…"})
        vault_now = load_vault(uid)

        # ── CAPTURE-ON-CONSUME ────────────────────────────────────────────
        # Resolve whichever intake step was left pending from the previous
        # turn BEFORE anything else runs, so a real value is guaranteed into
        # its vault field regardless of whether the free-text extractor
        # below manages to parse this same message. This is what fixes
        # fields staying permanently empty: a step is never abandoned just
        # because its question was displayed — only once it's answered.
        pending_key = vault_now.get("intake_pending")
        if pending_key:
            self._capture_intake_answer(pending_key, user_input, vault_now, uid)
            vault_now["intake_pending"] = ""
            save_vault(uid, vault_now)

        if not _looks_like_confirmation(user_input):
            profile_changed = self._quick_extract_and_merge(uid, user_input, vault_now)
            if profile_changed:
                # Push what we just learned to the client immediately, so the
                # Profile panel updates this turn instead of waiting on the
                # slower /api/extract-profile-from-chat pass after the reply.
                yield _sse({"profile_patch": {k: vault_now.get(k) for k in
                    ("name", "sector", "target_role", "location", "salary_range",
                     "work_type", "skills", "goals", "hobbies", "values",
                     "culture_preference",
                     "headline", "experience", "cv_summary", "experience_years")}})

        # ── DETERMINISTIC INTAKE SEQUENCE (mirrors the Jack & Jill reference
        # flow) — this runs BEFORE intent classification/persona selection so
        # it is always the thing driving the conversation until the brief is
        # complete, exactly like the reference: even an explicit "search for
        # roles" request gets intercepted here until we actually know enough
        # to search well. Skipped only mid-crisis, so distress always gets a
        # human response first — the missing field just gets asked next turn.
        lower_now = user_input.lower()
        is_distressed_now = (
            any(k in lower_now for k in self._DISTRESS_KEYWORDS)
            or vault_now.get("learning", {}).get("current_emotion") in ("frustrated", "anxious", "stuck")
        )
        if not is_distressed_now and not _looks_like_confirmation(user_input):
            step = self._next_intake_step(vault_now)
            # Edge case: target_role hasn't been asked/pending yet at all
            # (e.g. this is the very first message) but it already contains
            # a discovery-trigger phrase — skip straight past it rather than
            # asking a question the user just told us they can't answer.
            if step and step["key"] == "target_role" and any(t in lower_now for t in self._ROLE_DISCOVERY_TRIGGERS):
                vault_now.setdefault("intake_skipped", []).append("target_role")
                save_vault(uid, vault_now)
                step = self._next_intake_step(vault_now)
            if step:
                yield _sse({"status": "One quick thing first…"})
                yield _sse(self._intake_step_response(step, vault_now, uid))
                return

        ctx = self._build_context(uid, user_input)

        intent  = self._classify_intent(user_input, ctx)
        persona = self._select_persona(user_input, ctx)
        lower   = user_input.lower()

        # ── PREFERENCE UPDATE (silent, before responding) ────────────────

        if intent == "preferences":
            prompt = ("Extract career preferences from: '" + user_input + "'\n"
                      "Return ONLY JSON: {\"sector\":\"\",\"target_role\":\"\",\"goals\":[],\"skills\":[]}")
            raw = query_ai(prompt)
            try:
                prefs = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
                vault = ctx["vault"]
                if prefs.get("sector"):      vault["sector"]      = prefs["sector"]
                if prefs.get("target_role"): vault["target_role"] = prefs["target_role"]
                if prefs.get("goals"):       vault["goals"]       = prefs["goals"]
                if prefs.get("skills"):
                    vault["skills"] = list(set(vault.get("skills",[]) + prefs["skills"]))
                save_vault(uid, vault)
                store_vector(uid, f"Preferences updated: {json.dumps(prefs)}", "preferences", 8)
                ctx = self._build_context(uid, user_input)  # refresh
            except: pass
            yield _sse({"status": "✓ Profile updated"})

        # ── JOB SEARCH — real, specific, live jobs (Fix #2, #5) ─────────
        if intent == "job_search":
            target  = ctx["target_role"] or ctx["current_bg"] or "sustainability"
            sector  = ctx["current_bg"] or "green economy"
            skills  = ctx["skills"]
            location = ctx.get("location") or "London"

            yield _sse({"status": f"🔍 Finding real '{target}' jobs near {location}…"})
            raw_results = find_specific_jobs(target, sector, skills, location=location,
                                              salary_range=ctx.get("salary_range",""),
                                              work_type=ctx.get("work_type",""),
                                              limitations=ctx.get("limitations",[]))

            yield _sse({"status": f"⚡ Found {len(raw_results)} roles — personalising…"})

            # Build cards with full profile for personalisation (Fix #5)
            job_cards = [build_job_card(r, ctx["vault"]) for r in raw_results[:5]]

            # Transition-aware narrative (Fix #6)
            transition = ""
            if ctx["current_bg"] and ctx["target_role"] and ctx["current_bg"] != ctx["target_role"]:
                transition = (f"CRITICAL: {ctx['name'] or 'This person'} is transitioning FROM {ctx['current_bg']} "
                              f"INTO {ctx['target_role']}. They do NOT yet work in {ctx['target_role']}. "
                              f"Present these roles as exciting opportunities to transition into, not as matching their current job.")

            results_txt = "\n".join(("- " + r.get("title","") + ": " + r.get("url","")) for r in raw_results)

            profile_txt = f"Background: {ctx['current_bg']}. Goal: {ctx['target_role']}. Skills: {', '.join(_safe_str_list(ctx['skills'])[:4])}."

            match_notes = "\n".join(
                f"- {c['title']}: {c['match_label']}" + (f" (missing: {', '.join(c['missing_skills'])})" if c.get("missing_skills") else "")
                for c in job_cards if c.get("match_label")
            )
            salary_info = salary_reality_check(job_cards)

            # Pull recent history so repeated "find jobs" requests don't produce
            # a near-identical canned response every time (previously this
            # prompt had no memory of prior turns at all).
            history_lines2 = []
            for m in ctx["recent_msgs"][-6:]:
                r = m.get("role",""); c = m.get("content","")[:250]
                if r == "user":       history_lines2.append(f"User: {c}")
                elif r == "assistant": history_lines2.append(f"Garden AI: {c}")
            history_str2 = "\n".join(history_lines2) if history_lines2 else "This is the first time job search has come up."

            prompt = (
                "You are a career consultant.\n"
                + profile_txt + "\n"
                + transition + "\n\n"
                + "RECENT CONVERSATION (do not repeat the same opening line, framing, or bullet "
                + "structure you already used here — if you already said 'let's break down your next "
                + "steps' or similar, say something different this time):\n"
                + history_str2 + "\n\n"
                + "Live job results:\n"
                + results_txt + "\n\n"
                + (f"Match assessment against their known skills/gaps:\n{match_notes}\n\n" if match_notes else "")
                + (f"Salary reality check: {salary_info['summary']}\n\n" if salary_info else "")
                + "Write 2 warm paragraphs: (1) acknowledge their specific transition journey — briefly, "
                + "and only if you haven't just said the same thing, "
                + "(2) highlight 2-3 most relevant roles and WHY they fit, referencing match level and any skill gaps honestly. "
                + ("Mention the salary reality check briefly. " if salary_info else "")
                + "Be specific and personal. Use Markdown links. "
                + "Do not add any commentary about these instructions in your reply."
            )

            narrative = query_ai(prompt)
            yield _sse({"status": "✓ Results ready"})
            yield _sse({
                "response":    narrative,
                "job_cards":   job_cards,
                "salary_insight": salary_info,
                "suggestions": [
                    "Help me tailor my CV for one of these",
                    "What skills do I need for these roles?",
                    "Generate a 30-day plan to get there",
                    "Prep me for an interview for one of these",
                ]
            })
            return

        # ── ROADMAP ───────────────────────────────────────────────────────
        if intent == "roadmap":
            yield _sse({"status": "📋 Assessing your skills gap…"})
            result = gen_roadmap(uid, user_input)
            yield _sse({"status": "✓ Roadmap ready"})
            name = ctx["name"]
            phases = result.get("phase_titles", [])
            phase_line = " → ".join(phases) if phases else ""
            response_text = (
                f"{'Here you go, ' + name + '! ' if name else ''}"
                f"{result.get('narrative_summary','')}\n\n"
                f"**{len(phases)} phases over ~{result.get('total_estimated_weeks','?')} weeks**"
                f"{(': ' + phase_line) if phase_line else ''}, with "
                f"{result.get('milestone_count', 0)} concrete milestones. "
                f"Each one is now tracked in **My Path** — tick them off as you go, "
                f"and I'll follow up on where you're at."
            )
            yield _sse({
                "response": response_text,
                "path": result.get("path"),
                "suggestions": ["What should I focus on first?","Find roles in this area","Show me my skill gaps"]
            })
            return

        # ── INTERVIEW PREP — real questions, grounded in a tracked posting ──
        if intent == "coaching" and any(k in lower for k in
                ["interview","mock interview","practice interview","prep for the interview","prep me"]):
            yield _sse({"status": "🎤 Building interview questions for you…"})
            # If they named/tracked a specific role, ground questions in it.
            tracked = ctx.get("tracked_jobs", [])
            job_match = None
            for j in tracked:
                if j.get("title","").lower() and j.get("title","").lower() in lower:
                    job_match = j; break
            if not job_match and tracked:
                job_match = tracked[0]  # most recently tracked role
            session = generate_mock_interview(uid, job_match)
            qs = session.get("questions", [])
            if not qs:
                yield _sse({"response": "I had trouble generating questions just now — mind trying again in a moment?"})
                return
            q_lines = "\n".join(f"**{i+1}. {q['question']}**\n*What a strong answer covers:* {q['what_good_looks_like']}"
                                for i, q in enumerate(qs))
            grounding = f" for **{session['job_title']}**" if job_match else f" for **{ctx.get('target_role') or 'this type of role'}**"
            response_text = (
                f"Here are 6 realistic interview questions{grounding}. Try answering one here in chat "
                f"and I'll give you specific feedback on it:\n\n{q_lines}"
            )
            yield _sse({
                "response": response_text,
                "interview_session_id": session["id"],
                "suggestions": ["My answer to Q1 is …", "Give me 3 more questions", "Help me structure a STAR answer"]
            })
            return

        # ── TRANSFERABLE SKILLS — "does my background count?" ──────────────
        if intent == "transferable_skills":
            yield _sse({"status": "🔍 Mapping what transfers from your background…"})
            mappings = map_transferable_skills(uid)
            if not mappings:
                yield _sse({
                    "response": (
                        "I'd love to map this properly, but I need a bit more to go on first — "
                        "could you tell me about your current or previous role, or upload your CV? "
                        "Once I know what you've actually done, I can tell you honestly what carries over."
                    ),
                    "suggestions": ["Upload my CV", "My background is …", "What sustainability roles fit me?"]
                })
                return
            lines = "\n".join(f"- **{m.get('skill','')}** (from: {m.get('source','')}): {m.get('relevance','')}"
                              for m in mappings[:6])
            target = ctx.get("target_role") or "sustainability roles"
            response_text = (
                f"Here's what genuinely carries over toward **{target}** — I only included things "
                f"with a real, honest link, not a stretch:\n\n{lines}"
            )
            yield _sse({
                "response": response_text,
                "suggestions": ["Find jobs that value these skills", "What skills am I still missing?", "Build me a roadmap"]
            })
            return

        # ── CV HELP ───────────────────────────────────────────────────────
        if intent == "cv_help":
            yield _sse({"status": "📄 Reviewing your CV context…"})
            cv = ctx["cv_summary"]
            target = ctx["target_role"]
            prompt = self._build_prompt(user_input, ctx, {
                "name": "cv_specialist",
                "style": "precise and actionable",
                "strategy": (
                    "Give specific, actionable CV advice tailored to their exact background and target role. "
                    "If they haven't uploaded a CV, gently ask them to. "
                    "Reference their actual skills and target. Be concrete."
                )
            }, intent="cv_help")
            reply = query_ai(prompt)
            yield _sse({"status": "✓ CV advice ready"})
            yield _sse({
                "response": reply,
                "suggestions": ["Upload my CV for analysis","What keywords should I add?","Help me write a summary"]
            })
            return

        # ── LOCAL SEARCH — location-aware rich results ────────────────────────
        if intent == "local_search":
            loc = extract_location(user_input, ctx["vault"].get("location",""))
            lower2 = user_input.lower()
            stype = "volunteering"
            if any(k in lower2 for k in ["course","training","learn","class"]): stype = "courses"
            elif any(k in lower2 for k in ["event","meetup","conference","talk"]): stype = "events"
            elif any(k in lower2 for k in ["community","group","network"]): stype = "community"
            elif any(k in lower2 for k in ["webinar","online talk","panel"]): stype = "webinars"
            elif any(k in lower2 for k in ["job","role","vacancy","work","hiring"]): stype = "jobs"

            yield _sse({"status": f"Searching {stype} near {loc}..."})
            results = rich_local_search(user_input, loc, stype,
                                         target_role=ctx.get("target_role",""),
                                         sector=ctx.get("current_bg",""),
                                         limitations=ctx.get("limitations",[]),
                                         work_type=ctx.get("work_type",""))
            yield _sse({"status": f"Found {len(results)} options — building your guide..."})

            target = ctx["target_role"] or "sustainability"
            name   = ctx["name"] or ""
            profile_line = ""
            if ctx["current_bg"] and ctx["target_role"] and ctx["current_bg"] != ctx["target_role"]:
                profile_line = f"They are transitioning FROM {ctx['current_bg']} INTO {ctx['target_role']}."
            elif ctx["target_role"]:
                profile_line = f"Target role: {ctx['target_role']}."

            results_block = "\n".join(
                f"- [{r['title']}]({r['url']}): {r['body'][:300]}"
                for r in results if r.get("url")
            )

            prompt = (
                "You are a warm, expert career companion helping someone navigate their sustainability career.\n"
                + (f"User's name: {name}. " if name else "")
                + profile_line + "\n\n"
                + f"They asked: \"{user_input}\"\n"
                + f"Location: {loc}\n\n"
                + "REAL SEARCH RESULTS (use these URLs exactly — never invent links):\n"
                + results_block + "\n\n"
                + "Write a rich, structured response:\n"
                + "1. One warm sentence acknowledging what they're looking for\n"
                + "2. For each of 3-4 relevant results:\n"
                + "   - Organisation name as clickable link [Name](URL)\n"
                + "   - What they do and why it's relevant to this person's goals (2-3 sentences)\n"
                + "   - 3-4 specific ways to get involved as bullet points\n"
                + "   - HOW to reach out: exact steps (email/form/phone), what to say\n"
                + "3. A 'Next Steps' section:\n"
                + "   - Step 1: How to contact (what to write)\n"
                + "   - Step 2: What to do if no reply after 7 days\n"
                + "   - Step 3: How this experience maps to their CV and target role\n"
                + "4. End with ONE open question to understand what they need next\n\n"
                + "Use Markdown. Be specific, warm, and practical. No generic advice. "
                + f"Connect everything to their goal: {target}."
            )

            reply = query_ai(prompt)
            yield _sse({"status": "Guide ready"})
            # Save resources to vault so they persist (Fix #1)
            vault2 = load_vault(uid)
            sr = vault2.setdefault("saved_resources", [])
            for r in results:
                if r.get("url") and not any(x.get("url")==r.get("url") for x in sr):
                    sr.insert(0, {
                        "id":       str(uuid.uuid4()),
                        "url":      r["url"],
                        "title":    r.get("title",""),
                        "body":     r.get("body","")[:200],
                        "type":     stype,
                        "location": loc,
                        "status":   "not_started",
                        "saved_at": datetime.now().isoformat()
                    })
            vault2["saved_resources"] = sr[:100]
            save_vault(uid, vault2)

            # Build tailored suggestions based on context
            suggs = [
                f"Help me write an outreach email to one of these",
                f"What skills will I gain from {stype} in {loc}?",
                f"Find {target} jobs near {loc}",
                "Build me a 30-day action plan around this"
            ]
            yield _sse({
                "response": reply,
                "map_location": loc,
                "map_results": results[:4],
                "suggestions": suggs
            })
            return

        # ── RESOURCES — real, live, tailored (Fix #1) ────────────────────
        if intent == "resources":
            target = ctx["target_role"] or ctx["current_bg"] or "sustainability"
            sector = ctx["current_bg"] or "green economy"
            location = ctx.get("location") or "London"
            yield _sse({"status": f"🔗 Finding live resources for {target}…"})
            resources = find_tailored_resources(target, sector, location,
                                                 limitations=ctx.get("limitations",[]),
                                                 work_type=ctx.get("work_type",""))
            resource_cards = build_resource_cards(resources[:10])
            prompt = self._build_prompt(user_input, ctx, {
                "name": "advisor",
                "style": "helpful and specific",
                "strategy": (
                    "Present tailored resources from the list provided, covering as many "
                    "different types as are available (courses, volunteering, jobs, events, "
                    "webinars, community groups) — not just one category. "
                    "Do NOT write out the links yourself or format them as Markdown — the actual "
                    "clickable/saveable resource cards are shown separately below your message, so "
                    "just narrate what's there and why it's a good fit in plain prose (name the "
                    "resource, don't paste its URL)."
                )
            }, resources=resources[:10], intent="resources")
            reply = query_ai(prompt)
            yield _sse({"status": "✓ Resources found"})

            # Persist so they show up in the Resources tab, not just in-chat (Fix #7).
            vault3 = load_vault(uid)
            sr3 = vault3.setdefault("saved_resources", [])
            for r in resources:
                if r.get("url") and not any(x.get("url")==r.get("url") for x in sr3):
                    sr3.insert(0, {
                        "id": str(uuid.uuid4()), "url": r["url"], "title": r.get("title",""),
                        "body": r.get("body","")[:200], "type": r.get("type","resource"),
                        "location": location, "status": "not_started",
                        "saved_at": datetime.now().isoformat()
                    })
            vault3["saved_resources"] = sr3[:100]
            save_vault(uid, vault3)

            yield _sse({
                "response": reply,
                "resource_cards": resource_cards,
                "suggestions": ["Find me jobs in this area","Build me a roadmap","What other resources exist?"]
            })
            return

        # ── CONVERSATIONAL / COACHING / GENERAL (Fix #2, #7, #8) ─────────
        yield _sse({"status": "🧠 Thinking…"})

        # For coaching, also pull resources if relevant
        extra_resources = []
        if intent == "coaching" or persona.get("should_add_resources"):
            target = ctx["target_role"] or ctx["current_bg"]
            if target:
                extra_resources = find_tailored_resources(
                    target, ctx["current_bg"] or "sustainability", ctx.get("location") or "London",
                    limitations=ctx.get("limitations",[]), work_type=ctx.get("work_type","")
                )[:3]
        extra_resource_cards = build_resource_cards(extra_resources)

        prompt = self._build_prompt(user_input, ctx, persona,
                                    resources=extra_resources if extra_resources else None,
                                    intent=intent)
        if extra_resources:
            prompt += ("\nThe resources above are shown separately as clickable/saveable cards — "
                       "just refer to them by name in your reply, don't paste their URLs or "
                       "format them as Markdown links yourself.")
        stuck_note = self._check_stuck_resources(uid, user_input)
        if stuck_note:
            prompt += stuck_note

        nudge_suggestions = []
        reply  = query_ai(prompt)

        # Generate personalised suggestions (Fix #7) — unless we just asked a
        # specific onboarding question, in which case its own answer pills
        # are more useful than generic LLM-invented follow-ups.
        if nudge_suggestions:
            suggestions = nudge_suggestions
        else:
            sugg_raw = query_ai(
                "User said: '" + user_input[:200] + "'\n"
                + "Mood: " + ctx["emotion"] + "\n"
                + "Target: " + (ctx["target_role"] or "unknown") + "\n"
                + "Generate 3 SHORT follow-up suggestions (max 6 words each). JSON array only: [\"s1\",\"s2\",\"s3\"]"
            )
            try:
                s = sugg_raw.find("["); e = sugg_raw.rfind("]")+1
                suggestions = json.loads(sugg_raw[s:e]) if s > -1 else []
            except:
                suggestions = ["Tell me more","Help me plan next steps","Find me relevant jobs"]

        yield _sse({"status": "✓ Response ready"})
        yield _sse({"response": reply, "resource_cards": extra_resource_cards, "suggestions": suggestions})

orchestrator = Orchestrator()

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE LEARNING ENGINE
# Runs after every user/assistant exchange and updates vault["learning"]
# ─────────────────────────────────────────────────────────────────────────────

LEARN_PROMPT = """Analyse this conversation exchange and extract learning signals.
Return ONLY valid JSON, no other text.

Exchange:
User: {user_msg}
Assistant: {assistant_msg}

User profile so far:
Current role: {current_role}
Target role: {target_role}
Known blockers: {blockers}

Return this JSON:
{{
  "emotion": "one of: excited|frustrated|anxious|confident|stuck|grateful|neutral",
  "needs_encouragement": true or false,
  "is_celebrating": true or false,
  "is_venting": true or false,
  "wants_action": true or false,
  "new_skills_mentioned": ["list of any skills user mentioned"],
  "new_blockers": ["any new obstacles or fears the user revealed"],
  "milestones_achieved": ["any achievements user mentioned e.g. finished a course"],
  "preferred_tone": "one of: warm|direct|detailed|brief|motivational|analytical",
  "key_insight": "one sentence summary of what you learned about this user"
}}"""

def learn_from_exchange(uid: str, user_msg: str, assistant_msg: str):
    """Extract learning signals from a message pair and update vault."""
    vault = load_vault(uid)
    learning = vault.setdefault("learning", {
        "emotion_history": [],
        "blockers": [],
        "preferred_tone": "warm",
        "response_depth": "balanced",
        "milestones_mentioned": [],
        "total_exchanges": 0,
        "encouragement_count": 0,
        "key_insights": []
    })

    prompt = LEARN_PROMPT.format(
        user_msg      = user_msg[:500],
        assistant_msg = assistant_msg[:300],
        current_role  = vault.get("sector", "unknown"),
        target_role   = vault.get("target_role", "unknown"),
        blockers      = ", ".join(_safe_str_list(learning.get("blockers"))[:5]) or "none known"
    )

    raw = query_ai(prompt)
    try:
        # Parse JSON robustly
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return
        signals = json.loads(raw[start:end])
    except Exception as e:
        logging.warning(f"learn_from_exchange parse error: {e}")
        return

    # Update emotion history (keep last 20)
    emotion = signals.get("emotion", "neutral")
    learning.setdefault("emotion_history", []).append({
        "emotion": emotion, "ts": datetime.now().isoformat()
    })
    learning["emotion_history"] = learning["emotion_history"][-20:]
    learning["current_emotion"] = emotion

    # Update tone preference if detected
    if signals.get("preferred_tone"):
        learning["preferred_tone"] = signals["preferred_tone"]

    # Accumulate blockers (deduplicated)
    for b in signals.get("new_blockers", []):
        if b and b not in learning.get("blockers", []):
            learning.setdefault("blockers", []).append(b)
    learning["blockers"] = learning.get("blockers", [])[-10:]

    # Track encouragement needs
    if signals.get("needs_encouragement"):
        learning["encouragement_count"] = learning.get("encouragement_count", 0) + 1

    # Store key insight in vector memory
    insight = signals.get("key_insight", "")
    if insight:
        learning.setdefault("key_insights", []).append(insight)
        learning["key_insights"] = learning["key_insights"][-10:]
        store_vector(uid, f"User insight: {insight}", "learning", importance=7)

    # Skills mentioned — add to vault skills, with source attribution
    for sk in signals.get("new_skills_mentioned", []):
        if sk:
            _add_skill(vault, sk, "chat", user_msg)

    # Milestones mentioned — trigger milestone award
    for m in signals.get("milestones_achieved", []):
        if m:
            award_milestone(uid, m, vault, requires_verification=True)
            learning.setdefault("milestones_mentioned", []).append(m)

    # Celebrations / venting — store for context
    if signals.get("is_celebrating"):
        store_vector(uid, f"User celebrated: {user_msg[:200]}", "celebration", 6)
    if signals.get("is_venting"):
        store_vector(uid, f"User vented: {user_msg[:200]}", "emotional", 6)

    learning["total_exchanges"] = learning.get("total_exchanges", 0) + 1
    vault["learning"] = learning
    save_vault(uid, vault)


def adapt_system_prompt(uid: str) -> str:
    """Build a dynamic, personalised system prompt from learned signals."""
    vault    = load_vault(uid)
    learning = vault.get("learning", {})

    emotion   = learning.get("current_emotion", "neutral")
    tone      = learning.get("preferred_tone", "warm")
    blockers  = learning.get("blockers", [])
    insights  = learning.get("key_insights", [])
    xp        = vault.get("journey", {}).get("xp", 0)
    target    = vault.get("target_role", "")
    sector    = vault.get("sector", "sustainability")
    name      = vault.get("name", "")

    # Base identity
    base = (
        "You are Garden AI — a warm, empathetic, expert sustainability career companion. "
        "You are simultaneously a work coach, a friend, a motivator, and a strategic advisor. "
        "You listen deeply, remember everything, and adapt to the person in front of you.\n\n"
        "Your core mission: support this person's journey from where they are now "
        f"to their goal{': ' + target if target else ''}. "
        "Every small step matters. Celebrate wins. Acknowledge struggles. Never judge.\n\n"
    )

    # Persona from learned signals
    persona = f"Speak in a {tone}, human tone. "
    if emotion in ["frustrated", "anxious", "stuck"]:
        persona += (
            "This person is currently feeling " + emotion + ". "
            "Lead with empathy and acknowledgement BEFORE any advice. "
            "Be gentle, reassuring, and remind them that progress isn't linear. "
        )
    elif emotion in ["excited", "confident"]:
        persona += (
            "This person is energised right now. "
            "Match their energy — be enthusiastic, forward-looking, action-oriented. "
        )
    elif emotion == "grateful":
        persona += "They're in a reflective, grateful mood. Be warm and forward-looking. "

    # Known context
    context_parts = []
    if name:     context_parts.append(f"Their name is {name}.")
    if sector:   context_parts.append(f"They work in or are transitioning to {sector}.")
    if target:   context_parts.append(f"Their target role is {target}.")
    if blockers: context_parts.append(f"Known blockers: {', '.join(_safe_str_list(blockers)[:3])}.")
    if insights: context_parts.append(f"Key things you know about them: {'. '.join(insights[-3:])}.")
    if xp > 0:   context_parts.append(f"They've earned {xp} XP on their journey — acknowledge progress when relevant.")

    context_block = "\n".join(context_parts)

    rules = (
        "\n\nRULES:\n"
        "- Always personalise. Never give generic advice.\n"
        "- Use London-specific resources, volunteering, and free courses when recommending actions.\n"
        "- When someone shares a win (however small), celebrate it before moving on.\n"
        "- When someone is stuck, first ask what they've tried before suggesting solutions.\n"
        "- Mention skills gap progress naturally when relevant.\n"
        "- Format responses with Markdown. Keep paragraphs short and scannable.\n"
        "- End with one clear, specific next step or question — never leave them without direction.\n"
    )

    return base + persona + "\n\n" + context_block + rules


# ─────────────────────────────────────────────────────────────────────────────
# SKILLS GAP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

JOURNEY_DEFAULTS = {
    "xp": 0,
    "level": 1,
    "current_role": "",
    "target_role": "",
    "assessed": False,
    "progress_pct": 0,
    "skills_gaps": [],      # [{skill, current_level(0-5), target_level(0-5), status, resources}]
    "milestones": [],       # [{id, title, type, xp, earned_at, icon}]
    "weekly_checkins": [],  # [{date, mood, note}]
    "streak_days": 0,
    "last_active_date": "",
    "last_checkin_date": ""
}

XP_REWARDS = {
    "cv_uploaded":       50,
    "profile_complete":  30,
    "skill_started":     20,
    "skill_progressed":  40,
    "skill_complete":    100,
    "job_tracked":       15,
    "roadmap_generated": 60,
    "coaching_session":  25,
    "streak_3":          50,
    "streak_7":          150,
    "milestone_custom":  75,
}

BADGES = {
    "first_step":    {"icon": "🌱", "title": "First Step",     "desc": "Started your journey"},
    "skill_master":  {"icon": "⚡", "title": "Skill Spark",    "desc": "Completed your first skill"},
    "job_hunter":    {"icon": "🎯", "title": "Job Hunter",     "desc": "Tracked 5 roles"},
    "road_builder":  {"icon": "🗺️", "title": "Road Builder",   "desc": "Generated a career roadmap"},
    "week_warrior":  {"icon": "🔥", "title": "Week Warrior",   "desc": "7-day activity streak"},
    "cv_hero":       {"icon": "📄", "title": "CV Hero",        "desc": "Uploaded and analysed your CV"},
    "deep_thinker":  {"icon": "🧠", "title": "Deep Thinker",   "desc": "Completed a coaching session"},
    "trailblazer":   {"icon": "🏆", "title": "Trailblazer",    "desc": "Reached 500 XP"},
}

ASSESS_PROMPT = """You are a career transition expert.
Current role/background: {current_role}
Target role: {target_role}
Sector: {sector}
CV summary: {cv_summary}
User skills: {skills}

Identify the 6-8 most important skills needed for the target role.
For each skill, estimate:
- current_level: 0-5 (0=none, 5=expert) based on their background
- target_level: 0-5 required for the target role
- priority: high/medium/low

Do NOT invent resource links or URLs — leave "resources" as an empty list;
real, live resources are looked up separately.

Return ONLY valid JSON:
{{
  "skills_gaps": [
    {{
      "skill": "skill name",
      "current_level": 0,
      "target_level": 4,
      "gap": 4,
      "priority": "high",
      "status": "open",
      "why_matters": "one sentence",
      "resources": []
    }}
  ],
  "summary": "2-sentence narrative about the transition",
  "estimated_months": 6,
  "progress_pct": 15
}}"""

def _attach_real_resources_to_gaps(gaps: list, target_role: str, sector: str,
                                    location: str, limitations: list, work_type: str) -> list:
    """Replace any LLM-imagined resource links with real, live-verified ones
    found per skill (Fix #1, #7 — this was the main source of broken links)."""
    def _fetch(skill_name):
        found = rich_local_search(skill_name, location, "courses",
                                   target_role=target_role, sector=sector,
                                   limitations=limitations, work_type=work_type)
        real = [f for f in found if f.get("type") != "search_fallback"][:2]
        if not real:
            real = found[:1]  # google fallback, at minimum
        return [{"title": r.get("title",""), "url": r.get("url",""),
                 "type": r.get("type","courses")} for r in real]

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch, g["skill"]): g for g in gaps[:8]}
        for fut in as_completed(futures):
            g = futures[fut]
            try:
                g["resources"] = fut.result()
            except Exception:
                g["resources"] = [google_search_fallback(f"{g['skill']} {target_role} course")]
    return gaps

def assess_skills_gap(uid: str, current_role: str = "", target_role: str = "") -> dict:
    """Run LLM-based skills gap assessment, then attach real, verified,
    live resources per skill (never the LLM's invented links) and store
    result in vault. Fix #1, #6, #7."""
    vault = load_vault(uid)
    cr    = current_role or vault.get("sector", "")
    tr    = target_role  or vault.get("target_role", "")

    prompt = ASSESS_PROMPT.format(
        current_role = cr or "not specified",
        target_role  = tr or "sustainability role",
        sector       = vault.get("sector", ""),
        cv_summary   = vault.get("cv_summary", "no CV uploaded")[:400],
        skills       = ", ".join(_safe_str_list(vault.get("skills"))) or "not listed"
    )

    raw = query_ai(prompt)
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        data  = json.loads(raw[start:end])
    except Exception as e:
        logging.error(f"assess_skills_gap parse: {e}")
        data = {"skills_gaps": [], "summary": "Assessment could not be completed.", "progress_pct": 0}

    gaps = data.get("skills_gaps", [])
    if gaps:
        gaps = _attach_real_resources_to_gaps(
            gaps, tr or "sustainability role", cr or vault.get("sector",""),
            vault.get("location","London"), vault.get("limitations",[]), vault.get("work_type","")
        )
    data["skills_gaps"] = gaps

    # Merge into journey
    journey = vault.get("journey", dict(JOURNEY_DEFAULTS))
    journey["skills_gaps"]   = gaps
    journey["progress_pct"]  = data.get("progress_pct", 0)
    journey["assessed"]      = True
    journey["current_role"]  = cr
    journey["target_role"]   = tr
    journey["summary"]       = data.get("summary", "")
    journey["estimated_months"] = data.get("estimated_months", 6)
    vault["journey"] = journey
    if tr: vault["target_role"] = tr
    if cr: vault["sector"]      = cr
    save_vault(uid, vault)

    # Award XP for completing assessment
    award_xp(uid, "first_step", vault)
    store_vector(uid,
        f"Skills gap assessed. Current: {cr}. Target: {tr}. Gaps: "
        + ", ".join(g["skill"] for g in gaps),
        "assessment", importance=9)

    return data

def award_xp(uid: str, event_type: str, vault: dict = None) -> int:
    """Award XP for an event. Returns new total."""
    if vault is None: vault = load_vault(uid)
    journey = vault.setdefault("journey", dict(JOURNEY_DEFAULTS))
    xp_gain = XP_REWARDS.get(event_type, 10)
    journey["xp"] = journey.get("xp", 0) + xp_gain
    # Level up every 200 XP
    journey["level"] = max(1, journey["xp"] // 200 + 1)
    # Trailblazer badge at 500 XP
    if journey["xp"] >= 500:
        _grant_badge(uid, "trailblazer", vault)
    save_vault(uid, vault)
    return journey["xp"]

def award_milestone(uid: str, title: str, vault: dict = None, requires_verification: bool = False):
    """Record a milestone and grant XP. Milestones that claim a skill was
    mastered/closed require verification (evidence or manual confirm) before
    they're shown as confirmed — same workflow as skill gaps (Fix #3)."""
    if vault is None: vault = load_vault(uid)
    journey  = vault.setdefault("journey", dict(JOURNEY_DEFAULTS))
    existing = [m["title"] for m in journey.get("milestones", [])]
    if title in existing: return
    journey.setdefault("milestones", []).insert(0, {
        "id":       str(uuid.uuid4()),
        "title":    title,
        "type":     "achievement",
        "icon":     "🏅",
        "xp":       XP_REWARDS["milestone_custom"],
        "earned_at":datetime.now().isoformat(),
        "verification_status": "pending" if requires_verification else "auto",
        "verified": not requires_verification,
        "evidence": "",
    })
    save_vault(uid, vault)
    award_xp(uid, "milestone_custom", vault)

def _checkin_due(journey: dict) -> bool:
    """True once a week has passed since the last check-in — but only for
    users who actually have something to check in on (an assessment or
    milestones), so brand-new users aren't nagged before they've started."""
    if not (journey.get("assessed") or journey.get("milestones")):
        return False
    last = journey.get("last_checkin_date", "")
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except Exception:
        return True
    return (datetime.now() - last_dt).days >= 7

def _grant_badge(uid: str, badge_key: str, vault: dict):
    journey  = vault.setdefault("journey", dict(JOURNEY_DEFAULTS))
    milestones = journey.setdefault("milestones", [])
    if any(m.get("badge_key") == badge_key for m in milestones): return
    b = BADGES.get(badge_key, {})
    milestones.insert(0, {
        "id":        str(uuid.uuid4()),
        "title":     b.get("title", badge_key),
        "type":      "badge",
        "badge_key": badge_key,
        "icon":      b.get("icon", "🏅"),
        "desc":      b.get("desc", ""),
        "xp":        0,
        "earned_at": datetime.now().isoformat(),
        "verification_status": "auto",
        "verified":  True,
        "evidence":  "",
    })

def generate_mock_interview(uid: str, job: dict = None) -> dict:
    """Generate realistic interview questions grounded in an actual tracked
    posting when one is given, rather than generic sustainability questions."""
    vault  = load_vault(uid)
    target = vault.get("target_role", "") or "a sustainability role"
    if job:
        job_ctx = (f"Job: {job.get('title', target)} at {job.get('company','an organisation')}\n"
                   f"Posting details: {(job.get('snippet') or job.get('body',''))[:600]}")
    else:
        job_ctx = f"No specific posting tracked yet — generate role-specific (not generic) questions for: {target}"

    prompt = f"""You are a hiring manager for a sustainability/green-economy role in London,
preparing a candidate for a real interview.
{job_ctx}
Candidate background: {vault.get('cv_summary','') or vault.get('sector','not specified')}

Generate 6 realistic interview questions for THIS specific role/posting — a mix of behavioural,
technical/sector-knowledge, and motivation questions. For each, note briefly what a strong answer
would need to cover.

Return ONLY valid JSON:
{{"questions": [{{"question": "...", "what_good_looks_like": "one sentence"}}]}}"""

    raw = query_ai(prompt)
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        questions = data.get("questions", [])
    except Exception:
        questions = []

    session = {
        "id":         str(uuid.uuid4()),
        "job_title":  job.get("title", target) if job else target,
        "job_url":    job.get("url", "") if job else "",
        "questions":  questions,
        "created_at": datetime.now().isoformat(),
        "answers":    {},
    }
    vault.setdefault("interview_sessions", []).insert(0, session)
    vault["interview_sessions"] = vault["interview_sessions"][:20]
    save_vault(uid, vault)
    award_xp(uid, "coaching_session", vault)
    store_vector(uid, f"Mock interview prepped for: {session['job_title']}", "interview_prep", importance=7)
    return session

def feedback_on_answer(uid: str, session_id: str, question: str, answer: str) -> str:
    """Give specific, constructive feedback on a practice interview answer."""
    prompt = f"""You are an experienced sustainability-sector hiring manager giving interview feedback.
Question asked: {question}
Candidate's answer: {answer}

Give constructive, specific feedback in 3-4 sentences: what was strong, what's missing (e.g. STAR
structure, concrete metrics, sector-specific terminology), and one concrete way to improve the answer."""
    feedback = query_ai(prompt)
    vault = load_vault(uid)
    for s in vault.get("interview_sessions", []):
        if s.get("id") == session_id:
            s.setdefault("answers", {})[question] = {"answer": answer, "feedback": feedback}
            break
    save_vault(uid, vault)
    return feedback

def update_skill_progress(uid: str, skill_name: str, new_level: int, note: str = "") -> dict:
    """Update a skill's current_level and recalculate overall progress."""
    vault   = load_vault(uid)
    journey = vault.setdefault("journey", dict(JOURNEY_DEFAULTS))
    gaps    = journey.get("skills_gaps", [])
    updated = None
    for g in gaps:
        if g["skill"].lower() == skill_name.lower():
            old_level = g.get("current_level", 0)
            g["current_level"] = min(new_level, g["target_level"])
            g["gap"] = max(0, g["target_level"] - g["current_level"])
            if g["current_level"] >= g["target_level"]:
                g["status"] = "closed"
                award_xp(uid, "skill_complete", vault)
                award_milestone(uid, f"Mastered {g['skill']}", vault, requires_verification=True)
                _grant_badge(uid, "skill_master", vault)
            elif g["current_level"] > old_level:
                g["status"] = "in_progress"
                award_xp(uid, "skill_progressed", vault)
            if note:
                g.setdefault("notes", []).append({"note": note, "ts": datetime.now().isoformat()})
            updated = g
            break

    # Recalculate overall progress
    if gaps:
        total_gap = sum(g["target_level"] for g in gaps)
        filled    = sum(g["current_level"] for g in gaps)
        journey["progress_pct"] = round((filled / total_gap) * 100) if total_gap else 0

    journey["skills_gaps"] = gaps
    vault["journey"] = journey
    save_vault(uid, vault)
    store_vector(uid, f"Skill updated: {skill_name} now at level {new_level}. {note}", "progress", 7)
    return updated or {}


# ─────────────────────────────────────────────────────────────────────────────
# COACHING PROMPTS DATA
# ─────────────────────────────────────────────────────────────────────────────
COACHING_DATA = {
    "General": [
        {"icon":"route",     "title":"Map your transition",        "desc":"Where are you now and where do you want to go? Let's build your path.", "prompt":"Help me map my career transition into sustainability. I want to understand the steps involved."},
        {"icon":"psychology","title":"Work through career anxiety", "desc":"Feeling stuck, uncertain, or overwhelmed? Let's talk it through.",          "prompt":"I am feeling anxious and uncertain about my career change into sustainability. Can we talk?"},
        {"icon":"mood",      "title":"Recover from rejection",     "desc":"Didn't get the role? Turn this into your next move.",                       "prompt":"I just got rejected for a role and need help processing it and figuring out next steps"},
        {"icon":"celebration","title":"Celebrate and build on a win","desc":"Got an interview or offer? Make the most of it.",                          "prompt":"I have good news to share about my job search and want to plan next steps"},
        {"icon":"explore",   "title":"Career options exploration",  "desc":"Not sure which green career path fits you? Let's find out.",               "prompt":"Help me explore which sustainability career paths suit my background and interests"},
        {"icon":"support",   "title":"Motivation and momentum",    "desc":"Feeling unmotivated? Let's reconnect with your why.",                       "prompt":"I am losing motivation in my career transition. Can you help me reconnect with my goals?"},
    ],
    "CV": [
        {"icon":"description","title":"CV gap analysis",           "desc":"Upload your CV and we will find exactly what is missing for your target role.", "prompt":"Please analyse my CV for sustainability roles and tell me what needs improving", "action":"upload_cv"},
        {"icon":"edit",      "title":"Rewrite my personal summary","desc":"Your opening three lines make or break your application.",                   "prompt":"Help me rewrite my CV personal summary for a sustainability role"},
        {"icon":"search",    "title":"Missing keywords",           "desc":"ESG, net-zero, circular economy — are they visible?",                       "prompt":"What sustainability keywords should I add to my CV and where should they go?"},
        {"icon":"bar_chart", "title":"Quantify your impact",       "desc":"Turn job duties into results with numbers and outcomes.",                    "prompt":"Help me rewrite my CV bullet points to show quantifiable impact relevant to sustainability"},
    ],
    "Interview": [
        {"icon":"record_voice_over","title":"Mock interview",      "desc":"Practice with real questions tailored to your target role.",                 "prompt":"Give me a mock interview for the sustainability role I am targeting. Ask me questions."},
        {"icon":"lightbulb", "title":"STAR method coaching",       "desc":"Structure your examples so they land every time.",                           "prompt":"Coach me on how to use the STAR method for sustainability interview answers"},
        {"icon":"help",      "title":"Questions to ask",           "desc":"Stand out by asking the right questions at the end.",                        "prompt":"What are the best questions I can ask in a sustainability job interview?"},
    ],
    "Salary": [
        {"icon":"payments",  "title":"Salary benchmarks UK",       "desc":"What should you be earning in your target role in the UK?",                  "prompt":"What are typical salary ranges for sustainability roles in the UK in 2026?"},
        {"icon":"handshake", "title":"Negotiate your offer",       "desc":"Get what you deserve. Real scripts for the conversation.",                   "prompt":"Help me negotiate my salary offer for a sustainability role. Give me scripts and strategies."},
        {"icon":"trending_up","title":"Ask for a raise",           "desc":"Timing, approach, and what to say.",                                         "prompt":"How and when should I ask for a raise in my current role while transitioning to sustainability?"},
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES — PAGE SERVING
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve brand assets (logos, illustrations) from static/ subfolder."""
    static_dir = os.path.join(BASE_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    return send_from_directory(static_dir, filename)



@app.route("/")
def serve_landing(): return send_from_directory(BASE_DIR, "code.html")
@app.route("/dashboard")
def serve_dashboard(): return send_from_directory(BASE_DIR, "dashboard.html")
@app.route("/chat")
def serve_chat(): return send_from_directory(BASE_DIR, "chatbot.html")
@app.route("/strategist")
def serve_strategist(): return send_from_directory(BASE_DIR, "strategist.html")
@app.route("/roadmap")
def serve_roadmap(): return send_from_directory(BASE_DIR, "roadmap.html")
@app.route("/view-skill/<skill_name>/report")
def view_report(skill_name):
    r = send_from_directory(os.path.join(SKILLS_DIR, skill_name), "report.html")
    r.headers["Cache-Control"] = "no-store"
    return r
@app.route("/static/vault/<path:filename>")
def serve_vault_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static","vault"), filename)

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES — API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat_route():
    data       = request.json or {}
    user_input = data.get("text","").strip()
    uid        = data.get("user_id","anonymous")

    if not user_input or user_input == "INITIALIZE_GARDEN_SESSION":
        vault    = load_vault(uid)
        email    = vault.get("email","")
        name     = vault.get("name","") or (email.split("@")[0].title() if email else "")
        reports  = len(vault.get("reports",[]))
        msgs     = vault.get("messages",[])
        is_new   = len(msgs) == 0
        target   = vault.get("target_role","")
        sector   = vault.get("sector","")
        location = vault.get("location","")
        has_profile = bool(name and (target or sector))
        greeting_suggestions = []

        if is_new or not has_profile:
            greeting = (
                f"Hi{', ' + name if name else ''}! Welcome to Train Garden — "
                f"part work coach, part career strategist, part supportive friend for your "
                f"move into sustainability and green careers.\n\n"
                f"**So I can actually get you the right jobs and resources instead of generic ones, "
                f"tell me a bit about where you're starting from** — are you completely new to this, "
                f"switching in from another field, or already working in it and looking to go further?"
            )
            greeting_suggestions = [
                "I'm completely new to sustainability",
                "I'm switching careers into this field",
                "I already work in this field",
                "Upload my CV",
            ]
        else:
            ctx_parts = []
            if target: ctx_parts.append(f"your goal of becoming a {target}")
            if location: ctx_parts.append(f"opportunities near {location}")
            checkin_due = _checkin_due(vault.get("journey", {}))
            checkin_line = (
                "\n\nIt's also been about a week since your last check-in — how's progress going? "
                "Anything feel stuck, or worth celebrating?"
            ) if checkin_due else ""
            greeting = (
                f"Welcome back{', ' + name if name else ''}! Great to see you again.\n\n"
                f"{'I have been thinking about ' + ' and '.join(ctx_parts) + '. ' if ctx_parts else ''}"
                f"{'You have ' + str(reports) + ' saved reports in your vault. ' if reports else ''}"
                f"What would you like to work on today? I am here to help — whether it is job searching, "
                f"skill building, interview prep, or just talking through where you are at."
                f"{checkin_line}"
            )
            if checkin_due:
                greeting_suggestions = ["Things are going well!", "I feel a bit stuck", "Let's find new jobs"]
            elif not location:
                greeting_suggestions = ["Find jobs matching my profile", "Build me a roadmap", "I'm based in …"]
            else:
                greeting_suggestions = ["Find me jobs", "Check my skill gaps", "Help me prep for an interview"]

        return jsonify({"response": greeting, "report_count": reports,
                        "checkin_due": _checkin_due(vault.get("journey", {})),
                        "has_cv": bool(vault.get("cv_summary")),
                        "is_new_user": is_new,
                        "suggestions": greeting_suggestions,
                        "profile": {"target_role": target, "sector": sector,
                                    "name": name, "location": location,
                                    "goals": vault.get("goals",[])}})

    session_id = data.get("session_id", "default")
    log_message(uid, "user", user_input, session_id=session_id)

    def generate():
        final_text  = ""
        final_cards = []
        final_res_cards = []
        final_path  = None
        final_suggs = []
        is_intake_exchange = False
        try:
            for chunk in orchestrator.process(user_input, uid):
                yield chunk
                try:
                    p = json.loads(chunk[len("data: "):].strip())
                    if "response" in p:    final_text  = p["response"]
                    if "job_cards" in p:   final_cards = p["job_cards"]
                    if "resource_cards" in p: final_res_cards = p["resource_cards"]
                    if "path" in p:        final_path  = p["path"]
                    if "suggestions" in p: final_suggs = p["suggestions"]
                    if p.get("intake_step"): is_intake_exchange = True
                except: pass
        except Exception as e:
            import traceback; traceback.print_exc()
            yield _sse({"response": f"System error: {e}"})
        if final_text:
            log_message(uid, "assistant", final_text,
                        session_id=session_id,
                        job_cards=final_cards, report_path=final_path,
                        resource_cards=final_res_cards,
                        suggestions=final_suggs)
            # Run adaptive learning in background (non-blocking) — but never on
            # a deterministic intake exchange. The user's message there is a
            # scripted menu/chip answer ("Remote", "Early-stage startup", "I'm
            # not sure yet — help me explore"), not organic prose, and running
            # it through the emotion classifier is how a benign menu click has
            # previously been misread as genuine distress and left a sticky
            # current_emotion flag that then hijacked a later, unrelated turn.
            # Real distress during intake is still caught separately by the
            # keyword check that gates the intake sequence itself.
            if not is_intake_exchange:
                try:
                    learn_from_exchange(uid, user_input, final_text)
                except Exception as e:
                    logging.warning(f"learn_from_exchange: {e}")
            # Check streak
            try:
                v = load_vault(uid)
                j = v.setdefault("journey", dict(JOURNEY_DEFAULTS))
                today = datetime.now().date().isoformat()
                if j.get("last_active_date") != today:
                    yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
                    if j.get("last_active_date") == yesterday:
                        j["streak_days"] = j.get("streak_days", 0) + 1
                        if j["streak_days"] == 3:  award_xp(uid, "streak_3",  v); _grant_badge(uid, "week_warrior", v)
                        if j["streak_days"] == 7:  award_xp(uid, "streak_7",  v)
                    else:
                        j["streak_days"] = 1
                    j["last_active_date"] = today
                    v["journey"] = j
                    save_vault(uid, v)
            except Exception as e:
                logging.warning(f"streak: {e}")

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/get-messages", methods=["GET"])
def get_messages():
    uid        = request.args.get("user_id")
    session_id = request.args.get("session_id","default")
    if not uid: return jsonify({"messages":[]}), 400
    vault = load_vault(uid)
    # Support multiple chat sessions
    if session_id and session_id != "default":
        sessions = vault.get("chat_sessions", {})
        msgs = sessions.get(session_id, {}).get("messages", [])
    else:
        msgs = vault.get("messages", [])
    return jsonify({"messages": msgs[-100:], "session_id": session_id})

@app.route("/api/list-sessions", methods=["GET"])
def list_sessions():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"sessions":[]}), 400
    vault = load_vault(uid)
    result = [{"id": "default", "title": "Main Chat",
               "created_at": vault.get("first_seen",""),
               "message_count": len(vault.get("messages",[]))}]
    sessions = vault.get("chat_sessions", {})
    for sid, s in sessions.items():
        result.append({"id": sid, "title": s.get("title","Chat"),
                        "created_at": s.get("created_at",""),
                        "message_count": len(s.get("messages",[]))})
    result.sort(key=lambda x: x.get("created_at",""), reverse=True)
    return jsonify({"sessions": result})

@app.route("/api/create-session", methods=["POST"])
def create_session():
    data  = request.json or {}
    uid   = data.get("user_id")
    title = data.get("title","New Chat")
    if not uid: return jsonify({"success":False}), 400
    vault    = load_vault(uid)
    sessions = vault.setdefault("chat_sessions", {})
    sid      = str(uuid.uuid4())[:8]
    sessions[sid] = {"title": title, "messages": [], "created_at": datetime.now().isoformat()}
    save_vault(uid, vault)
    return jsonify({"success":True, "session_id": sid, "title": title})

@app.route("/api/delete-session", methods=["POST"])
def delete_session():
    data = request.json or {}
    uid  = data.get("user_id")
    sid  = data.get("session_id")
    if not uid or not sid: return jsonify({"success":False}), 400
    vault = load_vault(uid)
    vault.get("chat_sessions",{}).pop(sid, None)
    save_vault(uid, vault)
    return jsonify({"success":True})

@app.route("/api/get-profile", methods=["GET"])
def get_profile():
    uid = request.args.get("user_id")
    if not uid: return jsonify({}), 400
    vault = load_vault(uid)
    return jsonify({k: vault.get(k, {} if k=="skill_sources" else "") for k in
                    ["email","name","sector","experience_years","target_role","goals","skills",
                     "cv_summary","first_seen","location","dream_job","hobbies","values",
                     "culture_preference","skill_sources",
                     "salary_range","work_type","profile_complete","onboarding_done",
                     "headline","experience","how_heard","onboarding_seen"]})


@app.route("/api/save-aspirations", methods=["POST"])
def save_aspirations():
    data = request.json or {}
    uid  = data.get("user_id")
    if not uid: return jsonify({"success":False}), 400
    vault = load_vault(uid)
    for k in ["aspirations","hobbies","dream_job","values","salary_range","work_type","limitations"]:
        if k in data:
            v = data[k]
            vault[k] = _safe_str_list(v) if isinstance(v, list) else v
    save_vault(uid, vault)
    # Ask LLM to map hobbies to skills
    if data.get("hobbies"):
        hobbies_str = ", ".join(_safe_str_list(data["hobbies"]))
        prompt = (f"Map these hobbies to professional transferable skills relevant to sustainability careers: {hobbies_str}\n"
                  "Return JSON only: {{\"hobby_skills\": [{{\"hobby\": \"\", \"skills\": []}}]}}")
        raw = query_ai(prompt)
        try:
            start = raw.find("{"); end = raw.rfind("}")+1
            mapped = json.loads(raw[start:end])
            vault["hobby_skill_map"] = mapped.get("hobby_skills",[])
            save_vault(uid, vault)
        except: pass
    store_vector(uid, f"Aspirations updated: dream job={data.get('dream_job','')}. Hobbies={', '.join(_safe_str_list(data.get('hobbies')))}", "aspirations", 8)
    return jsonify({"success":True, "hobby_skills": vault.get("hobby_skill_map",[])})

@app.route("/api/get-aspirations", methods=["GET"])
def get_aspirations():
    uid = request.args.get("user_id")
    if not uid: return jsonify({}), 400
    vault = load_vault(uid)
    return jsonify({k: vault.get(k,"") for k in
                    ["aspirations","hobbies","dream_job","values","salary_range",
                     "work_type","limitations","hobby_skill_map"]})

@app.route("/api/save-profile", methods=["POST"])
def save_profile():
    data = request.json or {}
    uid  = data.get("user_id")
    if not uid: return jsonify({"success":False}), 400
    vault = load_vault(uid)
    for k in ["name","sector","experience_years","target_role","goals","skills","headline"]:
        if k in data: vault[k] = data[k]
    if isinstance(data.get("experience"), list):
        cleaned = []
        for e in data["experience"]:
            if not isinstance(e, dict):
                continue
            title = str(e.get("title","")).strip(); company = str(e.get("company","")).strip()
            if not (title or company):
                continue
            cleaned.append({
                "id": e.get("id") or uuid.uuid4().hex[:8],
                "title": title, "company": company,
                "start_date": str(e.get("start_date","")).strip(),
                "end_date":   str(e.get("end_date","")).strip(),
                "description": str(e.get("description","")).strip(),
                "source": e.get("source","manual"),
            })
        vault["experience"] = cleaned
    save_vault(uid, vault)
    store_vector(uid, f"Profile updated: {json.dumps({k:data[k] for k in data if k!='user_id'})}", "profile", 9)
    return jsonify({"success":True})

@app.route("/api/onboarding", methods=["POST"])
def onboarding():
    """Save the pre-chat onboarding step (name + how they heard about us) and
    mark it complete so it never shows again for this user."""
    data = request.json or {}
    uid  = data.get("user_id")
    if not uid: return jsonify({"success":False}), 400
    vault = load_vault(uid)
    if data.get("name"):
        vault["name"] = str(data["name"]).strip()
    if data.get("how_heard"):
        vault["how_heard"] = str(data["how_heard"]).strip()
    vault["onboarding_seen"] = True
    save_vault(uid, vault)
    return jsonify({"success": True})

@app.route("/api/get-jobs", methods=["GET"])
def get_jobs():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"jobs":[]}), 400
    vault = load_vault(uid)
    return jsonify({"jobs": vault.get("tracked_jobs",[])})

@app.route("/api/get-checkins", methods=["GET"])
def get_checkins():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"checkins": []}), 400
    vault = load_vault(uid)
    return jsonify({"checkins": vault.get("journey", {}).get("weekly_checkins", []),
                    "due": _checkin_due(vault.get("journey", {}))})

@app.route("/api/mock-interview", methods=["POST"])
def mock_interview_route():
    data    = request.json or {}
    uid     = data.get("user_id")
    job_url = data.get("job_url", "")
    if not uid: return jsonify({"success": False}), 400
    job = None
    if job_url:
        vault = load_vault(uid)
        job = next((j for j in vault.get("tracked_jobs", []) if j.get("url") == job_url), None)
    session = generate_mock_interview(uid, job)
    return jsonify({"success": True, "session": session})

@app.route("/api/interview-feedback", methods=["POST"])
def interview_feedback_route():
    data       = request.json or {}
    uid        = data.get("user_id")
    session_id = data.get("session_id", "")
    question   = data.get("question", "")
    answer     = data.get("answer", "")
    if not uid or not question or not answer: return jsonify({"success": False}), 400
    feedback = feedback_on_answer(uid, session_id, question, answer)
    return jsonify({"success": True, "feedback": feedback})

@app.route("/api/get-interview-sessions", methods=["GET"])
def get_interview_sessions():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"sessions": []}), 400
    vault = load_vault(uid)
    return jsonify({"sessions": vault.get("interview_sessions", [])})

@app.route("/api/map-transferable-skills", methods=["POST"])
def map_transferable_skills_route():
    data = request.json or {}
    uid  = data.get("user_id")
    if not uid: return jsonify({"success": False}), 400
    mappings = map_transferable_skills(uid)
    return jsonify({"success": True, "mappings": mappings})

@app.route("/api/get-transferable-skills", methods=["GET"])
def get_transferable_skills():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"mappings": []}), 400
    vault = load_vault(uid)
    return jsonify({"mappings": vault.get("transferable_skills", [])})

@app.route("/api/track-job", methods=["POST"])
def track_job_route():
    data = request.json or {}
    uid  = data.get("user_id")
    job  = data.get("job",{})
    if not uid or not job: return jsonify({"success":False}), 400
    added = track_job(uid, job)
    return jsonify({"success":True, "tracked": added})

@app.route("/api/update-job-status", methods=["POST"])
def update_job_status():
    data   = request.json or {}
    uid    = data.get("user_id")
    url    = data.get("url")
    status = data.get("status","interested")
    if not uid or not url: return jsonify({"success":False}), 400
    vault = load_vault(uid)
    for j in vault.get("tracked_jobs",[]):
        if j.get("url") == url: j["status"] = status
    save_vault(uid, vault)
    return jsonify({"success":True})

@app.route("/api/get-documents", methods=["GET"])
def get_documents():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"documents":[],"reports":[]}), 400
    vault = load_vault(uid)
    return jsonify({"documents": vault.get("documents",[]),
                    "reports":   vault.get("reports",[])})

@app.route("/api/get-history", methods=["GET"])
def get_history_api():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"history":[],"messages":[]}), 400
    vault = load_vault(uid)
    return jsonify({"history": vault.get("reports",[]),
                    "messages": vault.get("messages",[])[-50:]})

@app.route("/api/manage-history", methods=["POST"])
def manage_history():
    data = request.json or {}
    uid  = data.get("user_id")
    if not uid: return jsonify({"success":False}), 400
    vault   = load_vault(uid)
    reports = vault.get("reports",[])
    if data.get("action") == "delete":
        vault["reports"] = [r for r in reports if r.get("path") != data.get("path")]
    elif data.get("action") == "pin":
        for r in reports:
            if r.get("path") == data.get("path"): r["pinned"] = not r.get("pinned",False)
    save_vault(uid, vault)
    return jsonify({"success":True})

@app.route("/api/dashboard-stats", methods=["GET"])
def dashboard_stats():
    uid   = request.args.get("user_id")
    if not uid: return jsonify({"total_reports":0,"recent_activity":[]}), 400
    vault = load_vault(uid)
    return jsonify({"total_reports":   len(vault.get("reports",[])),
                    "pinned_count":    sum(1 for r in vault.get("reports",[]) if r.get("pinned")),
                    "recent_activity": vault.get("reports",[])[:3],
                    "has_cv":          bool(vault.get("cv_summary")),
                    "goals":           vault.get("goals",[])})

@app.route("/api/coaching-prompts", methods=["GET"])
def coaching_prompts():
    return jsonify(COACHING_DATA)

@app.route("/api/auth/request", methods=["POST"])
def handle_request_magic_code():
    email = (request.json or {}).get("email","").strip().lower()
    if not email: return jsonify({"success":False,"message":"Email required"}), 400
    code = str(random.randint(1000,9999))
    store_code(email, code)
    print(f"\n{'*'*40}\nMAGIC CODE for {email}: {code}\n{'*'*40}\n")
    try:
        msg = Message("Your Train Garden Access Code",
                      sender=app.config.get("MAIL_USERNAME"), recipients=[email])
        msg.body = f"Code: {code}\n(Expires in 5 minutes)"
        mail.send(msg)
    except Exception as e: print(f"Mail error: {e}")
    return jsonify({"success":True,"message":"Code sent"})

@app.route("/api/auth/verify", methods=["POST"])
def verify():
    data  = request.json or {}
    email = data.get("email","").strip().lower()
    code  = str(data.get("code","")).strip()
    if not check_code(email, code):
        return jsonify({"success":False,"message":"Invalid or expired code"}), 401
    consume_code(email)
    uid   = hashlib.md5(email.encode()).hexdigest()
    vault = load_vault(uid)
    if not vault.get("email"): vault["email"] = email
    vault["last_seen"] = datetime.now().isoformat()
    save_vault(uid, vault)
    return jsonify({"success":True,"user_id":uid,"email":email,
                    "report_count":  len(vault.get("reports",[])),
                    "has_cv":        bool(vault.get("cv_summary")),
                    "target_role":   vault.get("target_role",""),
                    "goals":         vault.get("goals",[])})

@app.route("/api/register", methods=["POST"])
def register():
    email = (request.json or {}).get("email","").strip().lower()
    uid   = hashlib.md5(email.encode()).hexdigest()
    return jsonify({"success":True,"user_id":uid})

@app.route("/api/upload-cv", methods=["POST"])
def upload_cv():
    try:
        file    = request.files.get("file")
        uid     = request.form.get("user_id","anonymous")
        job_ctx = request.form.get("job_description","sustainability role")
        if not file: return jsonify({"success":False,"message":"No file"}), 400

        path = os.path.join(COMPANION_DIR, f"{uid}_cv.pdf")
        file.save(path)
        cv_text = extract_pdf_text(path)
        if not cv_text.strip(): return jsonify({"success":False,"message":"Could not read PDF"}), 400

        prompt = (f"Analyse this CV for sustainability/green economy roles.\nContext: {job_ctx}\n"
                  f"CV:\n{cv_text[:3000]}\n\n"
                  "Provide: 1) SUMMARY (2-3 sentences), 2) MATCHED SKILLS, 3) SKILL GAPS, "
                  "4) TOP 3 NEXT STEPS\n\n"
                  "Then output every field you can find in the CV as: SKILLS_JSON: "
                  '{"matched":[],"gaps":[],"summary":"",'
                  '"name":"","sector":"","experience_years":"","suggested_target_role":"",'
                  '"location":"","goals":[],"hobbies":[],"headline":"",'
                  '"experience":[{"title":"","company":"","start_date":"","end_date":"","description":""}]}\n'
                  "location = city/area they live in or are based, if stated. "
                  "goals = any explicit career objective/summary statements. "
                  "hobbies = interests/hobbies section if present. "
                  "headline = their current/most senior job title as a short headline. "
                  "experience = every role in their work history section, most recent first, "
                  "with start_date/end_date as written on the CV (use 'Present' if ongoing). "
                  "Leave fields empty if not found — never invent.")
        raw = query_ai(prompt)
        # query_ai() never raises on failure (Ollama unreachable, etc.) — it
        # returns _AI_DOWN_FALLBACK, a plain sentence indistinguishable from a
        # normal reply at a glance. Detect that case explicitly so we (a) never
        # treat that sentence as CV analysis output, and (b) still record that
        # the CV itself was received, independent of whether the AI extraction
        # step worked. Previously, on an AI outage, `skills` silently stayed at
        # its all-empty defaults with no signal to the user or the rest of the
        # app that extraction had failed.
        ai_available = (raw != _AI_DOWN_FALLBACK)

        skills = {"matched":[],"gaps":[],"summary":"","name":"","sector":"","experience_years":"",
                  "suggested_target_role":"","location":"","goals":[],"hobbies":[],
                  "headline":"","experience":[]}
        md = raw
        if ai_available and "SKILLS_JSON:" in raw:
            parts = raw.split("SKILLS_JSON:")
            md    = parts[0].strip()
            try: skills = {**skills, **json.loads(parts[1].strip())}
            except: pass
        elif not ai_available:
            md = ("I've saved your CV, but I couldn't run the automatic analysis just now "
                  "because my AI model server isn't reachable. You can carry on describing "
                  "your background in chat in the meantime — I won't ask you for your CV "
                  "again, and I'll pick the analysis up automatically next time it's back.")

        # Store in vault — MERGE, never overwrite existing user-entered data.
        # The user can always edit/override any of these fields manually afterwards.
        # Pull in everything the CV can tell us about the profile (Fix #2).
        vault = load_vault(uid)
        # BUGFIX: `skills.get("summary", cv_text[:300])` never fell back to the
        # raw CV text, because "summary" is always *present* in `skills` (as ""
        # by default) — dict.get's default only applies when the key is
        # missing, not when it's empty. That meant cv_summary silently ended up
        # "" any time the AI didn't return a summary (including every AI outage),
        # which in turn made the onboarding "background" step think no CV/
        # background had ever been provided, and re-ask for it. Use `or` so an
        # empty/missing summary always falls back to the CV text itself.
        vault["cv_summary"] = skills.get("summary") or cv_text[:300]
        # Record unconditionally that a CV was received, regardless of whether
        # the AI extraction step succeeded — this is the authoritative signal
        # the onboarding intake should rely on, so a CV upload is never
        # re-requested just because analysis failed or found nothing.
        vault["cv_uploaded"] = True
        if skills.get("matched"):
            for sk in _safe_str_list(skills["matched"]):
                _add_skill(vault, sk, "cv", f"Matched from CV ({file.filename or 'uploaded document'})")
        if skills.get("goals"):
            vault["goals"] = list(set(vault.get("goals", []) + _safe_str_list(skills["goals"])))
        if skills.get("hobbies") and not vault.get("hobbies"):
            vault["hobbies"] = _safe_str_list(skills["hobbies"])
        if skills.get("name") and not vault.get("name"):
            vault["name"] = skills["name"]
        if skills.get("sector") and not vault.get("sector"):
            vault["sector"] = skills["sector"]
        if skills.get("experience_years") and not vault.get("experience_years"):
            vault["experience_years"] = skills["experience_years"]
        suggested_role = skills.get("suggested_target_role","") if not vault.get("target_role") else ""
        if skills.get("location") and not vault.get("location"):
            vault["location"] = skills["location"]
        if skills.get("headline") and not vault.get("headline"):
            vault["headline"] = skills["headline"]
        _upsert_experience(vault, skills.get("experience") or [])
        vault.setdefault("documents",[]).insert(0, {
            "id": str(uuid.uuid4()), "name": file.filename or "CV",
            "type":"cv", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        save_vault(uid, vault)
        store_vector(uid, f"CV: {vault['cv_summary']} | Skills: {', '.join(_safe_str_list(skills.get('matched')))} | Gaps: {', '.join(_safe_str_list(skills.get('gaps')))}", "cv", 9)

        # Immediately run the skills-gap assessment so gaps show up in "My Path"
        # right away — but only once we actually know what they're aiming for.
        # Running this against an empty target_role produces generic/misleading
        # gaps, so skip it until a target role or dream job is on file.
        gap_data = {}
        if vault.get("target_role") or vault.get("dream_job"):
            try:
                gap_data = assess_skills_gap(uid, vault.get("sector",""), vault.get("target_role","") or vault.get("dream_job",""))
            except Exception as e:
                logging.error(f"auto assess after CV upload: {e}")
        else:
            gap_data = {"skipped": True, "reason": "No target role or career goal on file yet — ask the user before assessing."}

        transferable = []
        try:
            transferable = map_transferable_skills(uid)
        except Exception as e:
            logging.error(f"auto transferable-skills after CV upload: {e}")

        # Only ever call this a "CV Gap Analysis" once a real, target-role-
        # grounded assessment has actually run. Labeling every CV upload that
        # way — even a bare "AI is down" apology (no analysis happened at
        # all), or a summary produced with no target role on file yet (so
        # assess_skills_gap was explicitly skipped above) — misrepresents
        # what the report actually contains.
        if not ai_available:
            report_title, report_type = "CV Received", "cv_upload"
        elif gap_data.get("skipped"):
            report_title, report_type = "CV Summary", "cv_summary"
        else:
            report_title, report_type = "CV Gap Analysis", "cv_analysis"

        # Build HTML report
        folder = f"analysis-{int(time.time())}"
        ws     = os.path.join(SKILLS_DIR, folder)
        os.makedirs(ws, exist_ok=True)
        html   = markdown.markdown(md, extensions=["tables"])
        page   = HTML_TMPL.replace("[[TITLE]]",report_title).replace("[[CONTENT]]", html)
        with open(os.path.join(ws,"report.html"),"w",encoding="utf-8") as f: f.write(page)
        p = f"/view-skill/{folder}/report"
        add_report(uid, report_title, p, report_type)

        # ── Keep the conversation moving instead of stopping at the report ──
        # A CV tells us what someone HAS done, never why they want to move —
        # so always turn this into a question, not a dead end.
        followups = []
        if suggested_role:
            followups.append(
                f"\n\nBased on your CV, **{suggested_role}** could be a strong target role — "
                f"want me to set that as your goal, or do you already have a different one in mind?"
            )
            suggestions = [f"Yes, set {suggested_role} as my target",
                           "I have a different role in mind",
                           "Help me explore options first"]
        elif not vault.get("target_role"):
            followups.append(
                "\n\nYour CV shows me where you've been — I'd still love to know **where you want to go**. "
                "What's the role or direction you're aiming for?"
            )
            suggestions = ["Help me explore which roles fit me",
                           "My target role is …",
                           "I'm not sure yet"]
        else:
            followups.append(f"\n\nThis fits with your goal of becoming a {vault['target_role']} — want me to find live roles or resources for that now?")
            suggestions = ["Find me jobs in this area", "Build me a roadmap", "What skills am I still missing?"]

        message = md + "".join(followups)

        return jsonify({"success":True,"message":message,"skills_data":skills,"path":p,
                        "suggested_target_role": suggested_role,
                        "suggestions": suggestions,
                        "gaps": gap_data.get("skills_gaps", []),
                        "transferable_skills": transferable})
    except Exception as e:
        logging.error(f"CV upload: {e}")
        return jsonify({"success":False,"message":str(e)}), 500

@app.route("/api/export-pdf", methods=["POST"])
def export_pdf():
    try:
        data    = request.json or {}
        title   = data.get("title","Report")
        content = data.get("content","").encode("latin-1","ignore").decode("latin-1")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica","B",16)
        pdf.cell(0,12,text=title.upper(),new_x="LMARGIN",new_y="NEXT",align="C")
        pdf.ln(8)
        pdf.set_font("helvetica",size=11)
        pdf.multi_cell(0,8,text=content)
        out = os.path.join(BASE_DIR,"latest_report.pdf")
        pdf.output(out)
        return send_file(out, as_attachment=True)
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500

@app.route("/journey")
def serve_journey():
    # My Path is now merged into the main Assistant chat interface
    return send_from_directory(BASE_DIR, "chatbot.html")

@app.route("/api/get-journey", methods=["GET"])
def get_journey():
    uid = request.args.get("user_id")
    if not uid: return jsonify({}), 400
    vault   = load_vault(uid)
    journey = vault.get("journey", dict(JOURNEY_DEFAULTS))
    return jsonify({
        "journey":    journey,
        "profile": {
            "name":        vault.get("name",""),
            "email":       vault.get("email",""),
            "target_role": vault.get("target_role",""),
            "sector":      vault.get("sector",""),
            "skills":      vault.get("skills",[]),
            "cv_summary":  vault.get("cv_summary","")
        },
        "learning":   vault.get("learning", {}),
        "streak":     journey.get("streak_days", 0),
        "xp":         journey.get("xp", 0),
        "level":      journey.get("level", 1)
    })

@app.route("/api/assess-skills", methods=["POST"])
def assess_skills():
    data         = request.json or {}
    uid          = data.get("user_id")
    current_role = data.get("current_role","")
    target_role  = data.get("target_role","")
    if not uid: return jsonify({"success": False}), 400

    vault = load_vault(uid)
    # Don't run a skills-gap assessment on a near-empty profile — it produces
    # generic/misleading results. Require: a name, a direction (target role or
    # dream job), and some background context (sector, CV, or experience).
    has_direction = bool((target_role or vault.get("target_role")) or vault.get("dream_job"))
    has_context   = bool((current_role or vault.get("sector")) or vault.get("cv_summary") or vault.get("experience_years"))
    if not (vault.get("name") and has_direction and has_context):
        missing = []
        if not vault.get("name"): missing.append("name")
        if not has_direction: missing.append("target role or career goal")
        if not has_context: missing.append("current background (sector, experience, or a CV upload)")
        return jsonify({
            "success": False,
            "message": "Not enough information yet to run a meaningful skills assessment.",
            "missing": missing
        }), 200

    result = assess_skills_gap(uid, current_role or vault.get("sector",""), target_role or vault.get("target_role",""))
    # Award XP for roadmap / CV upload events
    if vault.get("cv_summary"): award_xp(uid, "cv_uploaded", vault)
    return jsonify({"success": True, "data": result})

GAP_STATUSES = {"open", "pending", "in_progress", "closed"}

@app.route("/api/set-gap-status", methods=["POST"])
def set_gap_status():
    """Directly set a skills-gap's status (open/pending/in_progress/closed) and
    optionally log a check-in note against it — the interactive gap workflow."""
    data   = request.json or {}
    uid    = data.get("user_id")
    skill  = data.get("skill", "")
    status = data.get("status", "")
    note   = data.get("note", "")
    if not uid or not skill or status not in GAP_STATUSES:
        return jsonify({"success": False, "message": "Missing/invalid fields"}), 400

    vault   = load_vault(uid)
    journey = vault.setdefault("journey", dict(JOURNEY_DEFAULTS))
    gaps    = journey.get("skills_gaps", [])
    updated = None
    for g in gaps:
        if g.get("skill","").lower() == skill.lower():
            g["status"] = status
            if note:
                g.setdefault("checkins", []).append({
                    "note": note, "status": status, "ts": datetime.now().isoformat()
                })
            if status == "closed":
                g["current_level"] = g.get("target_level", g.get("current_level", 0))
                award_xp(uid, "skill_complete", vault)
                award_milestone(uid, f"Closed gap: {g['skill']}", vault, requires_verification=True)
            elif status == "in_progress":
                award_xp(uid, "skill_started", vault)
            updated = g
            break
    if updated is None:
        return jsonify({"success": False, "message": "Skill not found in gaps"}), 404

    journey["skills_gaps"] = gaps
    vault["journey"] = journey
    save_vault(uid, vault)
    store_vector(uid, f"Gap '{skill}' set to {status}. {note}", "gap_checkin", 7)
    return jsonify({"success": True, "gap": updated})

@app.route("/api/update-skill", methods=["POST"])
def update_skill():
    data       = request.json or {}
    uid        = data.get("user_id")
    skill_name = data.get("skill","")
    new_level  = int(data.get("level", 0))
    note       = data.get("note","")
    if not uid or not skill_name: return jsonify({"success": False}), 400
    updated = update_skill_progress(uid, skill_name, new_level, note)
    vault   = load_vault(uid)
    return jsonify({
        "success":    True,
        "skill":      updated,
        "xp":         vault.get("journey",{}).get("xp",0),
        "progress":   vault.get("journey",{}).get("progress_pct",0),
        "milestones": vault.get("journey",{}).get("milestones",[])[:3]
    })

@app.route("/api/add-evidence", methods=["POST"])
def add_evidence():
    """User reports progress in natural language — LLM extracts which skill improved."""
    data    = request.json or {}
    uid     = data.get("user_id")
    message = data.get("message","")
    if not uid or not message: return jsonify({"success":False}), 400

    vault = load_vault(uid)
    gaps  = vault.get("journey",{}).get("skills_gaps",[])
    skill_names = [g["skill"] for g in gaps]

    prompt = (f"The user said: '{message}'\n"
              f"Available skills to track: {skill_names}\n"
              "Which skill did they make progress on? What level (0-5) are they now? "
              "Return JSON only: {{\"skill\": \"skill name\", \"level\": 3, \"summary\": \"what they achieved\"}}")
    raw = query_ai(prompt)
    try:
        start = raw.find("{"); end = raw.rfind("}")+1
        ev    = json.loads(raw[start:end])
        sk    = ev.get("skill",""); lv = ev.get("level",0)
        if sk and sk in skill_names:
            updated = update_skill_progress(uid, sk, lv, ev.get("summary",""))
            vault2  = load_vault(uid)
            return jsonify({
                "success":  True,
                "detected_skill": sk,
                "new_level": lv,
                "xp":         vault2.get("journey",{}).get("xp",0),
                "progress":   vault2.get("journey",{}).get("progress_pct",0),
                "milestones": vault2.get("journey",{}).get("milestones",[])[:3]
            })
    except Exception as e:
        logging.warning(f"add_evidence: {e}")
    return jsonify({"success": False, "message": "Could not detect skill progress"})

@app.route("/api/checkin", methods=["POST"])
def weekly_checkin():
    data  = request.json or {}
    uid   = data.get("user_id")
    mood  = data.get("mood","neutral")   # great|good|okay|hard|stuck
    note  = data.get("note","")
    if not uid: return jsonify({"success":False}), 400
    vault = load_vault(uid)
    journey = vault.setdefault("journey", dict(JOURNEY_DEFAULTS))
    journey.setdefault("weekly_checkins",[]).insert(0,{
        "date": datetime.now().isoformat(), "mood": mood, "note": note
    })
    journey["weekly_checkins"] = journey["weekly_checkins"][:52]
    journey["current_mood"]    = mood
    journey["last_checkin_date"] = datetime.now().isoformat()
    vault["journey"] = journey
    save_vault(uid, vault)
    store_vector(uid, f"Mood check-in: {mood}. Note: {note}", "mood", 5)
    award_xp(uid, "coaching_session", vault)
    # Generate personalised response based on mood
    tone_map = {"great":"celebratory","good":"warm","okay":"steady","hard":"empathetic","stuck":"supportive"}
    prompt = (f"User mood: {mood}. Note: '{note}'\n"
              f"Context: {vault.get('target_role','')} career journey.\n"
              f"Give a {tone_map.get(mood,'warm')}, 2-sentence response. Acknowledge where they are. "
              "End with one tiny action they can take today.")
    reply = query_ai(prompt)
    return jsonify({"success":True, "response": reply})

@app.route("/api/get-learning", methods=["GET"])
def get_learning():
    uid = request.args.get("user_id")
    if not uid: return jsonify({}), 400
    vault = load_vault(uid)
    return jsonify({
        "learning":     vault.get("learning",{}),
        "emotion":      vault.get("learning",{}).get("current_emotion","neutral"),
        "tone":         vault.get("learning",{}).get("preferred_tone","warm"),
        "blockers":     vault.get("learning",{}).get("blockers",[]),
        "insights":     vault.get("learning",{}).get("key_insights",[]),
        "total_exchanges": vault.get("learning",{}).get("total_exchanges",0)
    })


# ─────────────────────────────────────────────────────────────────────────────
# SAVED ITEMS  (Fix #2 — save chat messages / resources to folders)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/save-item", methods=["POST"])
def save_item():
    data    = request.json or {}
    uid     = data.get("user_id")
    item    = data.get("item", {})   # {type, title, content, url, folder_id}
    if not uid or not item:
        return jsonify({"success": False}), 400
    vault   = load_vault(uid)
    item["id"]         = str(uuid.uuid4())
    item["saved_at"]   = datetime.now().isoformat()
    item.setdefault("folder_id", "default")
    vault.setdefault("saved_items", []).insert(0, item)
    vault["saved_items"] = vault["saved_items"][:500]
    save_vault(uid, vault)
    return jsonify({"success": True, "item_id": item["id"]})

@app.route("/api/get-saved", methods=["GET"])
def get_saved():
    uid       = request.args.get("user_id")
    folder_id = request.args.get("folder_id")   # optional filter
    if not uid: return jsonify({"items": [], "folders": []}), 400
    vault   = load_vault(uid)
    items   = vault.get("saved_items", [])
    if folder_id and folder_id != "all":
        items = [i for i in items if i.get("folder_id") == folder_id]
    return jsonify({"items": items, "folders": vault.get("folders", [])})

@app.route("/api/delete-saved", methods=["POST"])
def delete_saved():
    data    = request.json or {}
    uid     = data.get("user_id")
    item_id = data.get("item_id")
    if not uid or not item_id: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    vault["saved_items"] = [i for i in vault.get("saved_items", []) if i.get("id") != item_id]
    save_vault(uid, vault)
    return jsonify({"success": True})

@app.route("/api/create-folder", methods=["POST"])
def create_folder():
    data = request.json or {}
    uid  = data.get("user_id")
    name = data.get("name", "").strip()
    if not uid or not name: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    folder = {"id": str(uuid.uuid4()), "name": name,
              "created_at": datetime.now().isoformat(), "colour": data.get("colour", "#005d42")}
    vault.setdefault("folders", []).append(folder)
    save_vault(uid, vault)
    return jsonify({"success": True, "folder": folder})

@app.route("/api/delete-folder", methods=["POST"])
def delete_folder():
    data      = request.json or {}
    uid       = data.get("user_id")
    folder_id = data.get("folder_id")
    if not uid or not folder_id: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    vault["folders"]     = [f for f in vault.get("folders", []) if f.get("id") != folder_id]
    vault["saved_items"] = [i for i in vault.get("saved_items", []) if i.get("folder_id") != folder_id]
    save_vault(uid, vault)
    return jsonify({"success": True})

# ─────────────────────────────────────────────────────────────────────────────
# STICKY NOTES  (Fix #10)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/get-notes", methods=["GET"])
def get_notes():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"notes": []}), 400
    vault = load_vault(uid)
    return jsonify({"notes": vault.get("sticky_notes", [])})

@app.route("/api/save-note", methods=["POST"])
def save_note():
    data    = request.json or {}
    uid     = data.get("user_id")
    note_id = data.get("id")     # if editing existing
    content = data.get("content", "").strip()
    colour  = data.get("colour", "#fef3c7")
    if not uid: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    notes = vault.setdefault("sticky_notes", [])
    if note_id:
        for n in notes:
            if n.get("id") == note_id:
                n["content"]    = content
                n["colour"]     = colour
                n["updated_at"] = datetime.now().isoformat()
                break
    else:
        notes.insert(0, {"id": str(uuid.uuid4()), "content": content, "colour": colour,
                         "created_at": datetime.now().isoformat(),
                         "updated_at": datetime.now().isoformat()})
    vault["sticky_notes"] = notes[:50]
    save_vault(uid, vault)
    return jsonify({"success": True, "notes": vault["sticky_notes"]})

@app.route("/api/delete-note", methods=["POST"])
def delete_note():
    data    = request.json or {}
    uid     = data.get("user_id")
    note_id = data.get("id")
    if not uid or not note_id: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    vault["sticky_notes"] = [n for n in vault.get("sticky_notes", []) if n.get("id") != note_id]
    save_vault(uid, vault)
    return jsonify({"success": True})

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENTS — add + delete  (Fix #9)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/add-document", methods=["POST"])
def add_document():
    """Add a document record (metadata only, file upload handled by upload-cv)."""
    data = request.json or {}
    uid  = data.get("user_id")
    doc  = data.get("document", {})
    if not uid or not doc: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    doc["id"]         = str(uuid.uuid4())
    doc["added_at"]   = datetime.now().isoformat()
    vault.setdefault("documents", []).insert(0, doc)
    save_vault(uid, vault)
    return jsonify({"success": True, "doc": doc})

@app.route("/api/delete-document", methods=["POST"])
def delete_document():
    data   = request.json or {}
    uid    = data.get("user_id")
    doc_id = data.get("doc_id")
    if not uid or not doc_id: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    vault["documents"] = [d for d in vault.get("documents", []) if d.get("id") != doc_id]
    save_vault(uid, vault)
    return jsonify({"success": True})

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE LINK VALIDATOR + REPORT  (Fix #3)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/validate-link", methods=["POST"])
def validate_link():
    url = (request.json or {}).get("url", "")
    if not url: return jsonify({"valid": False}), 400
    try:
        r = requests.head(url, timeout=5, allow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0"})
        return jsonify({"valid": r.status_code < 400, "status": r.status_code})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})

@app.route("/api/report-link", methods=["POST"])
def report_link():
    data = request.json or {}
    uid  = data.get("user_id", "anonymous")
    url  = data.get("url", "")
    reason = data.get("reason", "broken")
    # Log to a simple file
    report_log = os.path.join(MEMORY_DIR, "link_reports.json")
    reports = []
    if os.path.exists(report_log):
        try:
            with open(report_log) as f: reports = json.load(f)
        except: pass
    reports.insert(0, {"url": url, "reason": reason, "uid": uid,
                        "ts": datetime.now().isoformat()})
    with open(report_log, "w") as f: json.dump(reports[:500], f)
    return jsonify({"success": True})

# ─────────────────────────────────────────────────────────────────────────────
# COMMUNITY  (Fix #6)
# ─────────────────────────────────────────────────────────────────────────────
COMMUNITY_DB = os.path.join(MEMORY_DIR, "community.json")

def load_community():
    if not os.path.exists(COMMUNITY_DB): return {"groups": [], "posts": []}
    try:
        with open(COMMUNITY_DB) as f: return json.load(f)
    except: return {"groups": [], "posts": []}

def save_community(data):
    with open(COMMUNITY_DB, "w") as f: json.dump(data, f, indent=2, ensure_ascii=False)

@app.route("/api/community/groups", methods=["GET"])
def get_groups():
    c = load_community()
    return jsonify({"groups": c.get("groups", [])})

@app.route("/api/community/create-group", methods=["POST"])
def create_group():
    data = request.json or {}
    uid  = data.get("user_id")
    name = data.get("name","").strip()
    desc = data.get("description","")
    tags = data.get("tags",[])
    if not uid or not name: return jsonify({"success": False}), 400
    c = load_community()
    group = {"id": str(uuid.uuid4()), "name": name, "description": desc,
             "tags": tags, "created_by": uid,
             "created_at": datetime.now().isoformat(),
             "members": [uid], "post_count": 0}
    c.setdefault("groups", []).insert(0, group)
    save_community(c)
    vault = load_vault(uid)
    vault.setdefault("community_groups", []).append(group["id"])
    save_vault(uid, vault)
    return jsonify({"success": True, "group": group})

@app.route("/api/community/join-group", methods=["POST"])
def join_group():
    data     = request.json or {}
    uid      = data.get("user_id")
    group_id = data.get("group_id")
    if not uid or not group_id: return jsonify({"success": False}), 400
    c = load_community()
    for g in c.get("groups", []):
        if g["id"] == group_id:
            if uid not in g.get("members", []):
                g.setdefault("members", []).append(uid)
    save_community(c)
    vault = load_vault(uid)
    vault.setdefault("community_groups", [])
    if group_id not in vault["community_groups"]:
        vault["community_groups"].append(group_id)
    save_vault(uid, vault)
    return jsonify({"success": True})

@app.route("/api/community/posts", methods=["GET"])
def get_posts():
    group_id = request.args.get("group_id")
    c        = load_community()
    posts    = c.get("posts", [])
    if group_id: posts = [p for p in posts if p.get("group_id") == group_id]
    return jsonify({"posts": posts[:50]})

@app.route("/api/community/post", methods=["POST"])
def create_post():
    data     = request.json or {}
    uid      = data.get("user_id")
    group_id = data.get("group_id")
    content  = data.get("content","").strip()
    title    = data.get("title","").strip()
    tags     = data.get("tags",[])
    if not uid or not content: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    name  = vault.get("name","") or vault.get("email","Anonymous").split("@")[0]
    c     = load_community()
    post  = {"id": str(uuid.uuid4()), "group_id": group_id, "user_id": uid,
             "author": name, "title": title, "content": content, "tags": tags,
             "created_at": datetime.now().isoformat(), "likes": 0, "replies": []}
    c.setdefault("posts", []).insert(0, post)
    # Update group post count
    for g in c.get("groups", []):
        if g["id"] == group_id: g["post_count"] = g.get("post_count", 0) + 1
    save_community(c)
    return jsonify({"success": True, "post": post})

@app.route("/api/community/reply", methods=["POST"])
def reply_post():
    data    = request.json or {}
    uid     = data.get("user_id")
    post_id = data.get("post_id")
    content = data.get("content","").strip()
    if not uid or not post_id or not content: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    name  = vault.get("name","") or vault.get("email","Anonymous").split("@")[0]
    c     = load_community()
    for p in c.get("posts", []):
        if p["id"] == post_id:
            p.setdefault("replies", []).append({
                "id": str(uuid.uuid4()), "user_id": uid, "author": name,
                "content": content, "created_at": datetime.now().isoformat()
            })
    save_community(c)
    return jsonify({"success": True})

@app.route("/api/community/like", methods=["POST"])
def like_post():
    data    = request.json or {}
    post_id = data.get("post_id")
    c       = load_community()
    for p in c.get("posts", []):
        if p["id"] == post_id:
            p["likes"] = p.get("likes", 0) + 1
    save_community(c)
    return jsonify({"success": True})

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCES  (Fix #5 — user can add/remove custom resources)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/add-resource", methods=["POST"])
def add_resource():
    data     = request.json or {}
    uid      = data.get("user_id")
    skill    = data.get("skill","")
    resource = data.get("resource",{})  # {title, url, type}
    if not uid or not resource: return jsonify({"success": False}), 400
    vault   = load_vault(uid)
    journey = vault.setdefault("journey", {})
    gaps    = journey.get("skills_gaps", [])
    for g in gaps:
        if not skill or g["skill"].lower() == skill.lower():
            g.setdefault("resources", []).append({
                "title": resource.get("title",""), "url": resource.get("url",""),
                "type":  resource.get("type","free_course"), "user_added": True,
                "added_at": datetime.now().isoformat()
            })
    journey["skills_gaps"] = gaps
    vault["journey"] = journey
    # Also save as a saved item
    vault.setdefault("saved_items", []).insert(0, {
        "id": str(uuid.uuid4()), "type": "resource",
        "title": resource.get("title",""), "url": resource.get("url",""),
        "folder_id": "resources", "saved_at": datetime.now().isoformat()
    })
    save_vault(uid, vault)
    return jsonify({"success": True})

@app.route("/api/remove-resource", methods=["POST"])
def remove_resource():
    data  = request.json or {}
    uid   = data.get("user_id")
    skill = data.get("skill","")
    url   = data.get("url","")
    if not uid: return jsonify({"success": False}), 400
    vault   = load_vault(uid)
    journey = vault.setdefault("journey", {})
    for g in journey.get("skills_gaps", []):
        if not skill or g["skill"].lower() == skill.lower():
            g["resources"] = [r for r in g.get("resources",[]) if r.get("url") != url]
    vault["journey"] = journey
    save_vault(uid, vault)
    return jsonify({"success": True})

# ─────────────────────────────────────────────────────────────────────────────
# DELETE from history/reports  (Fix #1 — delete buttons)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/delete-report", methods=["POST"])
def delete_report():
    data      = request.json or {}
    uid       = data.get("user_id")
    report_id = data.get("report_id")
    if not uid or not report_id: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    vault["reports"] = [r for r in vault.get("reports", [])
                        if r.get("id") != report_id and r.get("path") != report_id]
    save_vault(uid, vault)
    return jsonify({"success": True})

# ─────────────────────────────────────────────────────────────────────────────
# SAVED section for dashboard  (Fix #2)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/get-all-saved", methods=["GET"])
def get_all_saved():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"saved": [], "folders": []}), 400
    vault = load_vault(uid)
    return jsonify({
        "saved":   vault.get("saved_items", []),
        "folders": vault.get("folders", []),
        "reports": vault.get("reports", []),
        "notes":   vault.get("sticky_notes", [])
    })



# ─────────────────────────────────────────────────────────────────────────────
# STATIC ASSETS — serve uploaded logos and images
# ─────────────────────────────────────────────────────────────────────────────
import shutil as _shutil

_assets_dir = os.path.join(BASE_DIR, "static", "assets")
os.makedirs(_assets_dir, exist_ok=True)

# Copy uploaded brand images to static/assets on startup
_uploads_dir = os.path.join(os.path.dirname(__file__), "COMPANION")
for _fname in ["Group_29.png", "Group_30.png", "Group_31.png",
               "Group_33.png", "Group_35.png", "Group_36.png", "daryna_1.png"]:
    _src = os.path.join(os.path.dirname(__file__), _fname)
    if os.path.exists(_src):
        _shutil.copy2(_src, os.path.join(_assets_dir, _fname))

@app.route("/static/assets/<path:filename>")
def serve_asset(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static", "assets"), filename)




@app.route("/jobs")
def serve_jobs():
    return send_from_directory(BASE_DIR, "chatbot.html")   # job board is in chatbot

@app.route("/community")
def serve_community():
    return send_from_directory(BASE_DIR, "chatbot.html")

@app.route("/api/export-content", methods=["POST"])
def export_content():
    """Export any content as PDF, markdown, or txt."""
    data    = request.json or {}
    uid     = data.get("user_id","")
    fmt     = data.get("format","pdf")   # pdf | md | txt
    title   = data.get("title","Export")
    content = data.get("content","")

    if fmt == "md":
        out = os.path.join(BASE_DIR, "export.md")
        with open(out,"w",encoding="utf-8") as f:
            f.write(f"# {title}\n\n{content}")
        return send_file(out, as_attachment=True, download_name=f"{title.replace(' ','_')}.md")

    if fmt == "txt":
        out = os.path.join(BASE_DIR, "export.txt")
        with open(out,"w",encoding="utf-8") as f:
            import re as _re
            f.write(re.sub(r"[*#_\[\]\(\)]","",content))
        return send_file(out, as_attachment=True, download_name=f"{title.replace(' ','_')}.txt")

    # PDF (default)
    try:
        from fpdf import FPDF
        import re as _re
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica","B",16)
        safe_title = title.encode("latin-1","ignore").decode("latin-1")
        pdf.cell(0,12,text=safe_title,new_x="LMARGIN",new_y="NEXT",align="C")
        pdf.ln(8)
        pdf.set_font("helvetica",size=11)
        safe = _re.sub(r"[*#_\[\]\(\)\`]","",content).encode("latin-1","ignore").decode("latin-1")
        pdf.multi_cell(0,7,text=safe)
        out = os.path.join(BASE_DIR,"export.pdf")
        pdf.output(out)
        return send_file(out, as_attachment=True, download_name=f"{title.replace(' ','_')}.pdf")
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500

@app.route("/api/save-location", methods=["POST"])
def save_location():
    """Save user location preference for local search."""
    data    = request.json or {}
    uid     = data.get("user_id")
    loc     = data.get("location","").strip()
    if not uid: return jsonify({"success":False}), 400
    vault = load_vault(uid)
    vault["location"] = loc
    save_vault(uid, vault)
    return jsonify({"success":True})

@app.route("/api/get-map-resources", methods=["POST"])
def get_map_resources():
    """Return resources with lat/lng for map display."""
    data     = request.json or {}
    uid      = data.get("user_id","")
    location = data.get("location","London")
    rtype    = data.get("type","volunteering")
    vault    = load_vault(uid)
    loc      = vault.get("location","") or location

    results  = ddgs_search(f"{rtype} sustainability {loc} address contact", n=8)
    # Geocode using a simple search heuristic — return what we have with location context
    places   = []
    for r in results:
        places.append({
            "title": r.get("title",""),
            "url":   r.get("url",""),
            "body":  r.get("body","")[:200],
            "location": loc,
        })
    return jsonify({"places": places, "location": loc})

@app.route("/api/community/upload", methods=["POST"])
def community_upload():
    """Handle file uploads in community posts."""
    uid     = request.form.get("user_id","")
    post_id = request.form.get("post_id","")
    file    = request.files.get("file")
    if not file: return jsonify({"success":False,"message":"No file"}), 400
    upload_dir = os.path.join(BASE_DIR, "static", "community_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = f"{uid}_{uuid.uuid4().hex}_{file.filename}"
    path      = os.path.join(upload_dir, safe_name)
    file.save(path)
    url = f"/static/community/{safe_name}"
    return jsonify({"success":True, "url": url, "name": file.filename})

@app.route("/static/community/<path:filename>")
def serve_community_file(filename):
    return send_from_directory(os.path.join(BASE_DIR,"static","community_uploads"), filename)



@app.route("/api/get-saved-resources", methods=["GET"])
def get_saved_resources():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"resources": []}), 400
    vault = load_vault(uid)
    return jsonify({"resources": vault.get("saved_resources", [])})

@app.route("/api/delete-saved-resource", methods=["POST"])
def delete_saved_resource():
    data = request.json or {}
    uid  = data.get("user_id")
    rid  = data.get("resource_id")
    if not uid or not rid: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    vault["saved_resources"] = [r for r in vault.get("saved_resources",[]) if r.get("id") != rid]
    save_vault(uid, vault)
    return jsonify({"success": True})

@app.route("/api/save-resource", methods=["POST"])
def save_resource_endpoint():
    """Manually save a resource from frontend."""
    data = request.json or {}
    uid  = data.get("user_id")
    res  = data.get("resource", {})
    if not uid or not res: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    sr = vault.setdefault("saved_resources", [])
    if not any(x.get("url") == res.get("url") for x in sr):
        res["id"]       = str(uuid.uuid4())
        res["saved_at"] = datetime.now().isoformat()
        res["status"]   = res.get("status", "not_started")
        sr.insert(0, res)
        vault["saved_resources"] = sr[:100]
        save_vault(uid, vault)
    return jsonify({"success": True})

@app.route("/api/update-resource-status", methods=["POST"])
def update_resource_status():
    """Track a saved resource's progress: not_started | in_progress | completed."""
    data   = request.json or {}
    uid    = data.get("user_id")
    rid    = data.get("resource_id")
    status = data.get("status", "in_progress")
    note   = data.get("note", "")
    if not uid or not rid: return jsonify({"success": False}), 400
    vault = load_vault(uid)
    updated = None
    for r in vault.get("saved_resources", []):
        if r.get("id") == rid:
            r["status"] = status
            if note: r["progress_note"] = note
            if status == "completed":
                r["completed_at"] = datetime.now().isoformat()
                award_xp(uid, "skill_progressed", vault)
                vault.setdefault("documents", []).insert(0, {
                    "id": str(uuid.uuid4()),
                    "name": r.get("title", "Resource"),
                    "type": "completed_resource",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                award_milestone(uid, f"Completed: {r.get('title','a resource')}", vault)
            updated = r
            break
    save_vault(uid, vault)
    store_vector(uid, f"Resource '{updated.get('title','') if updated else rid}' marked {status}. {note}",
                 "resource_progress", 6)
    return jsonify({"success": True, "resource": updated})

# ── Skill Validation ──────────────────────────────────────────────────────────
@app.route("/api/validate-skill", methods=["POST"])
def validate_skill():
    """User uploads certification or marks skill as validated."""
    uid       = request.form.get("user_id","")
    skill     = request.form.get("skill","")
    status    = request.form.get("status","validated")  # validated | in_progress | needed
    cert_file = request.files.get("cert")
    if not uid or not skill: return jsonify({"success":False}), 400
    vault = load_vault(uid)
    sv    = vault.setdefault("skill_validations", {})
    sv[skill] = {"status": status, "updated_at": datetime.now().isoformat()}
    if cert_file:
        safe_skill = re.sub(r"[^a-z0-9]","-", skill.lower())[:30]
        stored_name = f"{uid}_{safe_skill}_{uuid.uuid4().hex[:8]}_{cert_file.filename}"
        cert_path = os.path.join(COMPANION_DIR, stored_name)
        cert_file.save(cert_path)
        sv[skill]["cert_file"] = cert_file.filename
        sv[skill]["cert_path"] = stored_name   # servable filename
    vault["skill_validations"] = sv
    save_vault(uid, vault)
    store_vector(uid, f"Skill {skill} marked as {status}", "skill_validation", 8)
    return jsonify({"success": True, "validations": sv})

@app.route("/static/certs/<uid>/<path:filename>")
def serve_cert(uid, filename):
    """Serve uploaded skill certificates so evidence is viewable."""
    return send_from_directory(COMPANION_DIR, filename)

@app.route("/api/verify-milestone", methods=["POST"])
def verify_milestone():
    """Confirm (or upload evidence for) a milestone that requires verification —
    e.g. 'Mastered X' / 'Closed gap: X'. Same click-to-expand-then-verify
    workflow as skill gaps (Fix #3)."""
    uid         = request.form.get("user_id","")
    milestone_id= request.form.get("milestone_id","")
    status      = request.form.get("status","verified")  # verified | pending | rejected
    note        = request.form.get("note","")
    evidence    = request.files.get("evidence")
    if not uid or not milestone_id: return jsonify({"success":False}), 400

    vault   = load_vault(uid)
    journey = vault.setdefault("journey", dict(JOURNEY_DEFAULTS))
    milestones = journey.get("milestones", [])
    updated = None
    for m in milestones:
        if m.get("id") == milestone_id:
            m["verification_status"] = status
            m["verified"] = (status == "verified")
            if note: m["verification_note"] = note
            if evidence:
                safe = re.sub(r"[^a-z0-9]","-", m.get("title","milestone").lower())[:30]
                stored_name = f"{uid}_{safe}_{uuid.uuid4().hex[:8]}_{evidence.filename}"
                evidence.save(os.path.join(COMPANION_DIR, stored_name))
                m["evidence"] = stored_name
                m["evidence_name"] = evidence.filename
            updated = m
            break
    if updated is None:
        return jsonify({"success": False, "message": "Milestone not found"}), 404

    journey["milestones"] = milestones
    vault["journey"] = journey
    save_vault(uid, vault)
    store_vector(uid, f"Milestone '{updated.get('title','')}' marked {status}. {note}", "milestone_verification", 7)
    return jsonify({"success": True, "milestone": updated})

@app.route("/static/milestone-evidence/<uid>/<path:filename>")
def serve_milestone_evidence(uid, filename):
    """Serve uploaded milestone evidence files."""
    return send_from_directory(COMPANION_DIR, filename)

@app.route("/api/get-skill-validations", methods=["GET"])
def get_skill_validations():
    uid = request.args.get("user_id")
    if not uid: return jsonify({"validations": {}}), 400
    vault = load_vault(uid)
    return jsonify({"validations": vault.get("skill_validations", {})})

# ── Profile via chat (agent fills profile from conversation) ──────────────────
@app.route("/api/extract-profile-from-chat", methods=["POST"])
def extract_profile_from_chat():
    """Extract profile data from the last N messages and update vault."""
    data = request.json or {}
    uid  = data.get("user_id")
    if not uid: return jsonify({"success":False}), 400
    vault = load_vault(uid)
    msgs  = vault.get("messages",[])[-20:]
    conv  = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in msgs)
    prompt = (
        "Extract profile information from this conversation.\n"
        f"Conversation:\n{conv}\n\n"
        "Return ONLY valid JSON with any fields found:\n"
        "{\"name\":\"\",\"target_role\":\"\",\"sector\":\"\","
        "\"experience_years\":\"\",\"goals\":[],\"skills\":[],"
        "\"hobbies\":[],\"dream_job\":\"\",\"values\":[],"
        "\"salary_range\":\"\",\"work_type\":\"\",\"location\":\"\","
        "\"headline\":\"\","
        "\"experience\":[{\"title\":\"\",\"company\":\"\",\"start_date\":\"\","
        "\"end_date\":\"\",\"description\":\"\"}]}\n"
        "\"headline\" = a short professional headline for their profile (e.g. "
        "'Generalist / AI Transformation Specialist'), only if inferable from what "
        "they've actually said about their work. "
        "\"experience\" = every distinct job/role they've described across the whole "
        "conversation, one entry per employer — including vague ones like 'stealth "
        "startup' with company left empty if no name was given. Never invent a company "
        "or dates that weren't mentioned; use 'Present' for an ongoing role's end_date."
    )
    raw = query_ai(prompt)
    updated = {}
    try:
        start = raw.find("{"); end = raw.rfind("}")+1
        extracted = json.loads(raw[start:end])
        if _upsert_experience(vault, extracted.get("experience") or []):
            updated["experience"] = vault["experience"]
        if extracted.get("headline") and not vault.get("headline"):
            vault["headline"] = extracted["headline"]; updated["headline"] = extracted["headline"]
        for k,v in extracted.items():
            if k in ("experience", "headline"):
                continue
            if v and v != "" and v != [] and k in vault:
                if isinstance(v, list) and isinstance(vault.get(k,[]), list):
                    existing = _safe_str_list(vault.get(k,[]))
                    merged = list(dict.fromkeys(existing + _safe_str_list(v)))
                    if merged != existing:
                        vault[k] = merged; updated[k] = merged
                elif isinstance(v, str) and v and not vault.get(k):
                    vault[k] = v; updated[k] = v
        if updated:
            save_vault(uid, vault)
            store_vector(uid, f"Profile auto-filled: {json.dumps(updated)}", "profile", 8)
    except: pass
    return jsonify({"success": True, "updated": updated})


@app.after_request
def add_headers(r):
    r.headers["X-Frame-Options"] = "SAMEORIGIN"
    return r

if __name__ == "__main__":
    app.run(debug=True, port=5001, threaded=True)