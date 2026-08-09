import os
import re
import json
import base64
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Dementia Memory Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = str(Path(__file__).resolve().parent)

DATA_DIR = os.path.join(BASE_DIR, "memory_data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_memory")
MEMORY_JSON_PATH = os.path.join(DATA_DIR, "memories.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

LOGO_PATH = "ai.png"

FALLBACK_MSG = "I do not remember that right now. Please ask your caregiver."
WELCOME_MSG = "Hello. I am here to help you remember people, objects, routines, and reminders."

MIN_RELEVANCE_SCORE = 0.35
MAX_RETRIEVED_DOCS = 4


# =========================================================
# BASIC HELPERS
# =========================================================

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""

    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except OSError:
        return ""


def escape_html(text):
    text = str(text)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


logo_base64 = get_base64_image(LOGO_PATH)


# =========================================================
# LIGHT CHATGPT-LIKE UI
# =========================================================

background_logo_css = ""

if logo_base64:
    background_logo_css = f"""
    .stApp::before {{
        content: "";
        position: fixed;
        top: 52%;
        left: 61%;
        width: 690px;
        height: 690px;
        transform: translate(-50%, -50%);
        background-image: url("data:image/png;base64,{logo_base64}");
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
        opacity: 0.035;
        pointer-events: none;
        z-index: 0;
    }}

    section[data-testid="stSidebar"]::after {{
        content: "";
        position: absolute;
        left: 50%;
        bottom: 3.5rem;
        width: 155px;
        height: 155px;
        transform: translateX(-50%);
        background-image: url("data:image/png;base64,{logo_base64}");
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
        opacity: 0.075;
        pointer-events: none;
        z-index: 0;
    }}
    """

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {{
        --app-bg: #F7F7F8;
        --sidebar-bg: #FFFFFF;
        --surface: #FFFFFF;
        --surface-soft: #F3F4F6;
        --assistant-bg: rgba(255, 255, 255, 0.96);
        --user-bg: #F3F4F6;
        --text: #111827;
        --muted: #6B7280;
        --border: #E5E7EB;
        --border-strong: #D1D5DB;
        --input-bg: #FFFFFF;
        --accent: #2563EB;
        --accent-soft: #EFF6FF;
        --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 12px 30px rgba(15,23,42,0.06);
    }}

    html, body, .stApp, .main, * {{
        font-family: 'Inter', ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
                     "Helvetica Neue", Arial, sans-serif !important;
    }}

    html, body, .stApp {{
        color: var(--text) !important;
        background: var(--app-bg) !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: 0rem !important;
        min-height: 0rem !important;
        display: none !important;
    }}

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {{
        display: none !important;
        visibility: hidden !important;
    }}

    .stApp,
    .main,
    div[data-testid="stAppViewContainer"],
    div[data-testid="stAppViewBlockContainer"] {{
        background: var(--app-bg) !important;
        color: var(--text) !important;
    }}

    div[data-testid="stAppViewBlockContainer"] {{
        padding-top: 2.2rem !important;
        position: relative !important;
        z-index: 1 !important;
    }}

    .block-container {{
        max-width: 1060px !important;
        padding-top: 2.2rem !important;
        padding-bottom: 7rem !important;
    }}

    {background_logo_css}

    section[data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border) !important;
        box-shadow: 2px 0 14px rgba(15,23,42,0.03) !important;
        position: relative !important;
    }}

    section[data-testid="stSidebar"] > div {{
        background: var(--sidebar-bg) !important;
        padding-top: 1.4rem !important;
        position: relative !important;
        z-index: 1 !important;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{
        color: var(--text) !important;
    }}

    .sidebar-logo-wrap {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0.35rem 0 1rem 0;
    }}

    .sidebar-logo {{
        width: 174px;
        height: 174px;
        object-fit: contain;
        display: block;
    }}

    .sidebar-title {{
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: var(--text);
        margin: 0.15rem 0 0.15rem 0;
        text-align: left;
    }}

    .sidebar-subtitle {{
        font-size: 0.88rem;
        color: var(--muted);
        line-height: 1.45;
        margin-bottom: 1.05rem;
    }}

    .sidebar-stat {{
        background: var(--surface-soft);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.8rem 0.85rem;
        margin: 0.75rem 0;
    }}

    .sidebar-stat-number {{
        color: var(--text);
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }}

    .sidebar-stat-label {{
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.1rem;
    }}

    .sidebar-help {{
        background: var(--accent-soft);
        border: 1px solid #DBEAFE;
        border-radius: 16px;
        padding: 0.8rem 0.85rem;
        color: #1E40AF;
        font-size: 0.83rem;
        line-height: 1.45;
        margin-top: 0.75rem;
    }}

    .hero-card {{
        background: rgba(255,255,255,0.84);
        border: 1px solid var(--border);
        border-radius: 26px;
        padding: 2rem 2.1rem;
        margin-bottom: 1.15rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(8px);
    }}

    .main-title {{
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.06em !important;
        margin-bottom: 0.55rem !important;
        color: var(--text) !important;
        line-height: 1.05 !important;
    }}

    .subtitle {{
        color: var(--muted) !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        margin-bottom: 0 !important;
        max-width: 760px !important;
    }}

    .quick-title {{
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0.9rem 0 0.45rem 0;
    }}

    .memory-card {{
        background-color: rgba(255,255,255,0.92);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.15rem 1.25rem;
        margin: 0.3rem 0 1rem 0;
        box-shadow: var(--shadow);
        backdrop-filter: blur(8px);
    }}

    .memory-card-title {{
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.4rem;
    }}

    .memory-card-body {{
        color: var(--text);
        line-height: 1.65;
        font-size: 0.97rem;
    }}

    .source-pill {{
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.22rem 0.6rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        color: var(--muted);
        background: var(--surface-soft);
        font-size: 0.82rem;
    }}

    div[data-testid="stChatMessage"] {{
        border-radius: 20px !important;
        border: 1px solid var(--border) !important;
        margin-bottom: 16px !important;
        padding: 0.5rem 0.65rem !important;
        box-shadow: var(--shadow) !important;
        line-height: 1.7 !important;
    }}

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
        background-color: var(--assistant-bg) !important;
    }}

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background-color: var(--user-bg) !important;
    }}

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li {{
        color: var(--text) !important;
        font-size: 1.03rem !important;
        line-height: 1.7 !important;
    }}

    div[data-testid="stChatInput"] {{
        background: transparent !important;
    }}

    div[data-testid="stChatInput"] textarea {{
        background-color: var(--input-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 16px !important;
        caret-color: var(--text) !important;
        min-height: 50px !important;
        box-shadow: 0 8px 28px rgba(15,23,42,0.08) !important;
    }}

    div[data-testid="stChatInput"] textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.10), 0 8px 28px rgba(15,23,42,0.08) !important;
    }}

    div[data-testid="stChatInput"] textarea::placeholder {{
        color: #9CA3AF !important;
        opacity: 1 !important;
    }}

    div.stButton > button {{
        background-color: transparent !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 999px !important;
        font-weight: 650 !important;
        padding: 0.48rem 0.95rem !important;
        text-align: left !important;
        box-shadow: none !important;
        transition: all 0.16s ease !important;
    }}

    div.stButton > button:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background-color: var(--accent-soft) !important;
    }}

    div.stButton > button:focus {{
        box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
        border-color: var(--accent) !important;
    }}

    hr {{
        border-color: var(--border) !important;
        opacity: 1 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MEMORY STORE / QUICK QUESTION HELPERS
# =========================================================

def load_memory_store():
    empty = {
        "people": [],
        "objects": [],
        "routines": [],
        "reminders": [],
        "documents": []
    }

    if not os.path.exists(MEMORY_JSON_PATH):
        return empty

    try:
        with open(MEMORY_JSON_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        for key in empty:
            if key not in data or not isinstance(data[key], list):
                data[key] = []

        return data

    except Exception:
        return empty


def build_quick_questions(memory_store):
    questions = []

    if memory_store.get("people"):
        name = memory_store["people"][0].get("name", "").strip()
        if name:
            questions.append(f"Who is {name}?")

    if memory_store.get("routines"):
        title = memory_store["routines"][0].get("title", "").strip()
        if title:
            questions.append(f"What is my {title} routine?")

    if memory_store.get("objects"):
        name = memory_store["objects"][0].get("name", "").strip()
        if name:
            questions.append(f"What is {name}?")

    if memory_store.get("reminders"):
        questions.append("What reminders do I have?")

    defaults = [
        "Who visits me often?",
        "What do I do in the morning?",
        "What medicine do I take?",
        "Who helps me with appointments?"
    ]

    for item in defaults:
        if len(questions) >= 4:
            break
        if item not in questions:
            questions.append(item)

    return questions[:4]


def queue_question(question):
    st.session_state.pending_question = question


# =========================================================
# RAG INITIALIZATION
# =========================================================

@st.cache_resource
def initialize_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    llm = Ollama(
        model="llama3.2",
        temperature=0.0,
        base_url="http://localhost:11434"
    )

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    return llm, vector_store


def retrieve_memories(vector_store, query):
    if vector_store is None:
        return []

    query = str(query).strip()

    if not query:
        return []

    try:
        results = vector_store.similarity_search_with_relevance_scores(
            query,
            k=MAX_RETRIEVED_DOCS
        )
    except Exception:
        return []

    accepted = []

    for document, score in results:
        if score < MIN_RELEVANCE_SCORE:
            continue

        accepted.append((document, score))

    accepted.sort(key=lambda item: item[1], reverse=True)

    return [
        document
        for document, _ in accepted
    ]


def build_context(documents):
    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "memory")
        memory_type = document.metadata.get("type", "memory")
        filename = document.metadata.get("filename", "")

        header = f"Memory {index} | Type: {memory_type} | Source: {source}"

        if filename:
            header += f" | File: {filename}"

        context_parts.append(
            f"{header}\n{document.page_content}"
        )

    return "\n\n".join(context_parts)


def clean_answer(text):
    text = str(text).replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_simple_memory_answer(user_question, documents):
    if not documents:
        return ""

    question = str(user_question).lower()
    top_document = documents[0]
    metadata = top_document.metadata
    content = top_document.page_content.strip()
    memory_type = metadata.get("type", "")

    if memory_type == "person" and any(word in question for word in ["who", "name", "person"]):
        name = metadata.get("name", "").strip()
        relationship = metadata.get("relationship", "").strip()

        if name and relationship:
            return f"This is {name}, your {relationship}. {content}"

    if memory_type in {"routine", "object", "reminder"}:
        return content

    return ""


def generate_answer(llm, user_question, context):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a gentle memory assistant for a person with dementia. "
            "Answer only using the provided caregiver memory context. "
            "Do not guess. Do not invent names, medical instructions, dates, or relationships. "
            "Keep the answer short, calm, and reassuring. "
            "Use simple language. "
            "If the context does not contain the answer, say exactly: "
            "'I do not remember that right now. Please ask your caregiver.'"
        ),
        (
            "human",
            "Caregiver memory context:\n{context}\n\nQuestion:\n{question}"
        )
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "context": context,
            "question": user_question
        })

        answer = clean_answer(response)

        if not answer:
            return FALLBACK_MSG

        unsafe_phrases = [
            "i think",
            "probably",
            "maybe",
            "usually, people",
            "in general",
            "as an ai",
            "based on general"
        ]

        lowered = answer.lower()

        if any(phrase in lowered for phrase in unsafe_phrases):
            return FALLBACK_MSG

        return answer

    except Exception:
        return FALLBACK_MSG


def render_memory_card(documents):
    if not documents:
        return

    top_document = documents[0]
    memory_type = escape_html(top_document.metadata.get("type", "memory"))
    source = escape_html(top_document.metadata.get("source", "memory"))
    filename = escape_html(top_document.metadata.get("filename", ""))

    source_line = source

    if filename:
        source_line += f" | {filename}"

    preview = escape_html(top_document.page_content.strip())

    if len(preview) > 430:
        preview = preview[:430].rstrip() + "..."

    st.markdown(
        f"""
        <div class="memory-card">
            <div class="memory-card-title">Retrieved Memory: {memory_type}</div>
            <div class="memory-card-body">
                {preview}
                <br>
                <span class="source-pill">Source: {source_line}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar_logo():
    if not logo_base64:
        return

    st.markdown(
        f"""
        <div class="sidebar-logo-wrap">
            <img src="data:image/png;base64,{logo_base64}" class="sidebar-logo">
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MSG
        }
    ]

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


memory_store = load_memory_store()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    render_sidebar_logo()

    st.markdown(
        """
        <div class="sidebar-title">Memory Assistant</div>
        <div class="sidebar-subtitle">
            Ask calm, simple questions using trusted caregiver memories.
        </div>
        """,
        unsafe_allow_html=True
    )

    total_sources = (
        len(memory_store.get("people", []))
        + len(memory_store.get("objects", []))
        + len(memory_store.get("routines", []))
        + len(memory_store.get("reminders", []))
        + len(memory_store.get("documents", []))
    )

    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-number">{total_sources}</div>
            <div class="sidebar-stat-label">Saved memory sources</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MSG
            }
        ]
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-help">
            This assistant only answers from saved caregiver knowledge. If it cannot find the answer, it will ask you to check with the caregiver.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MAIN APP
# =========================================================

llm, vector_store = initialize_rag()

st.markdown(
    """
    <div class="hero-card">
        <div class='main-title'>Dementia Memory Assistant</div>
        <div class='subtitle'>A calm assistant that answers only from trusted caregiver memories.</div>
    </div>
    """,
    unsafe_allow_html=True
)

quick_questions = build_quick_questions(memory_store)

if quick_questions:
    st.markdown("<div class='quick-title'>Quick questions</div>", unsafe_allow_html=True)
    cols = st.columns(2)

    for index, question in enumerate(quick_questions):
        with cols[index % 2]:
            st.button(
                question,
                key=f"quick_question_{index}",
                use_container_width=True,
                on_click=queue_question,
                args=(question,)
            )

st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.session_state.pop("pending_question", "")

if not user_question:
    user_question = st.chat_input(
        "Ask a memory question, for example: Who is Sarah?"
    )

if user_question:
    user_question = str(user_question).strip()

    if user_question:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        documents = retrieve_memories(vector_store, user_question)

        if documents:
            render_memory_card(documents)
            simple_answer = get_simple_memory_answer(user_question, documents)

            if simple_answer:
                answer = simple_answer
            else:
                context = build_context(documents)
                answer = generate_answer(llm, user_question, context)
        else:
            answer = FALLBACK_MSG

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
