import os
import re
import json
import uuid
import base64
from datetime import date
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Caregiver Memory Admin",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = str(Path(__file__).resolve().parent)

DATA_DIR = os.path.join(BASE_DIR, "memory_data")
PHOTO_DIR = os.path.join(DATA_DIR, "photos")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_memory")
UPLOAD_DIR = os.path.join(BASE_DIR, "caregiver_documents")

MEMORY_JSON_PATH = os.path.join(DATA_DIR, "memories.json")

LOGO_PATH = "ai.png"

for folder in [DATA_DIR, PHOTO_DIR, CHROMA_DIR, UPLOAD_DIR]:
    os.makedirs(folder, exist_ok=True)


# =========================================================
# UI STYLING
# Keeps same dark style/colors as your current chatbot direction.
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

background_logo_css = ""

if logo_base64:
    background_logo_css = f"""
    .stApp::before {{
        content: "";
        position: fixed;
        top: 52%;
        left: 61%;
        width: 680px;
        height: 680px;
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
        opacity: 0.08;
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
        --card-bg: rgba(255, 255, 255, 0.94);
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
        font-family: 'Inter', ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
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
        padding-top: 1.5rem !important;
        position: relative !important;
        z-index: 1 !important;
    }}

    .block-container {{
        max-width: 1120px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
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
        margin: 0.25rem 0 1rem 0;
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

    .sidebar-section-label {{
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 1rem 0 0.45rem 0;
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

    div[role="radiogroup"] label {{
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 12px !important;
        padding: 0.42rem 0.55rem !important;
        margin-bottom: 0.12rem !important;
        transition: all 0.15s ease !important;
    }}

    div[role="radiogroup"] label:hover {{
        background: var(--surface-soft) !important;
        border-color: var(--border) !important;
    }}

    .hero-card {{
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--border);
        border-radius: 26px;
        padding: 2rem 2.1rem;
        margin-bottom: 1.3rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(8px);
    }}

    .admin-title {{
        font-size: 2.65rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.06em !important;
        margin-bottom: 0.55rem !important;
        color: var(--text) !important;
        line-height: 1.05 !important;
    }}

    .admin-subtitle {{
        color: var(--muted) !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        margin-bottom: 0 !important;
        max-width: 760px !important;
    }}

    .memory-card {{
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(8px);
    }}

    .memory-type {{
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.35rem;
    }}

    .memory-name {{
        color: var(--text);
        font-size: 1.14rem;
        font-weight: 750;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
    }}

    .memory-meta {{
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }}

    .memory-notes {{
        color: var(--text);
        line-height: 1.65;
        font-size: 0.95rem;
        word-break: break-word;
    }}

    .demo-flow {{
        background: rgba(255,255,255,0.88);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.3rem 1.45rem;
        box-shadow: var(--shadow);
    }}

    .demo-flow h3 {{
        margin-top: 0 !important;
        color: var(--text) !important;
        letter-spacing: -0.03em !important;
    }}

    .demo-flow li {{
        margin-bottom: 0.45rem !important;
        line-height: 1.55 !important;
        color: var(--text) !important;
    }}

    .metric-card {{
        background: rgba(255,255,255,0.90);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.1rem 1.15rem;
        box-shadow: var(--shadow);
    }}

    .metric-number {{
        color: var(--text);
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.06em;
        line-height: 1;
    }}

    .metric-label {{
        color: var(--muted);
        font-size: 0.84rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }}

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stDateInput"] input,
    div[data-testid="stTimeInput"] input,
    div[data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }}

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.10) !important;
    }}

    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder {{
        color: #9CA3AF !important;
        opacity: 1 !important;
    }}

    section[data-testid="stFileUploader"] {{
        background: var(--surface) !important;
        border: 1px dashed var(--border-strong) !important;
        border-radius: 20px !important;
        padding: 1rem !important;
    }}

    section[data-testid="stFileUploader"] button {{
        background-color: transparent !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }}

    section[data-testid="stFileUploader"] button:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background-color: var(--accent-soft) !important;
    }}

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {{
        background-color: transparent !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 999px !important;
        font-weight: 650 !important;
        padding: 0.48rem 0.95rem !important;
        box-shadow: none !important;
        transition: all 0.16s ease !important;
    }}

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background-color: var(--accent-soft) !important;
    }}

    div.stButton > button:focus,
    div[data-testid="stFormSubmitButton"] button:focus {{
        box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
        border-color: var(--accent) !important;
    }}

    div[data-testid="stAlert"] {{
        border-radius: 16px !important;
        border: 1px solid var(--border) !important;
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
# JSON STORAGE
# =========================================================

def empty_store():
    return {
        "people": [],
        "objects": [],
        "routines": [],
        "reminders": [],
        "documents": []
    }


def load_store():
    if not os.path.exists(MEMORY_JSON_PATH):
        return empty_store()

    try:
        with open(MEMORY_JSON_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        base = empty_store()

        for key in base:
            if key not in data or not isinstance(data[key], list):
                data[key] = []

        return data

    except (OSError, json.JSONDecodeError):
        return empty_store()


def save_store(data):
    with open(MEMORY_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def safe_filename(filename):
    filename = os.path.basename(str(filename)).strip()
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def save_photo(uploaded_photo, prefix):
    if uploaded_photo is None:
        return ""

    extension = os.path.splitext(uploaded_photo.name)[1].lower()

    if extension not in [".png", ".jpg", ".jpeg", ".webp"]:
        extension = ".jpg"

    filename = f"{prefix}_{uuid.uuid4().hex}{extension}"
    path = os.path.join(PHOTO_DIR, filename)

    with open(path, "wb") as file:
        file.write(uploaded_photo.getbuffer())

    return path


def save_document_file(uploaded_file):
    filename = safe_filename(uploaded_file.name)
    path = os.path.join(UPLOAD_DIR, filename)

    base_name, extension = os.path.splitext(filename)
    counter = 1

    while os.path.exists(path):
        filename = f"{base_name}_{counter}{extension}"
        path = os.path.join(UPLOAD_DIR, filename)
        counter += 1

    with open(path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return path, filename


# =========================================================
# CHROMA HELPERS
# =========================================================

@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


def memory_to_document(memory_type, record):
    if memory_type == "people":
        content = (
            f"{record.get('name', '')} is {record.get('relationship', '')}. "
            f"Memory notes: {record.get('notes', '')}"
        )

        metadata = {
            "source": "admin_form",
            "type": "person",
            "name": record.get("name", ""),
            "relationship": record.get("relationship", ""),
            "record_id": record.get("id", "")
        }

    elif memory_type == "objects":
        content = (
            f"{record.get('name', '')} is an important object. "
            f"Description: {record.get('description', '')}"
        )

        metadata = {
            "source": "admin_form",
            "type": "object",
            "name": record.get("name", ""),
            "record_id": record.get("id", "")
        }

    elif memory_type == "routines":
        content = (
            f"{record.get('title', '')} routine at {record.get('time', '')}. "
            f"Steps and notes: {record.get('steps', '')}"
        )

        metadata = {
            "source": "admin_form",
            "type": "routine",
            "title": record.get("title", ""),
            "time": record.get("time", ""),
            "record_id": record.get("id", "")
        }

    elif memory_type == "reminders":
        content = (
            f"Reminder: {record.get('title', '')}. "
            f"Date: {record.get('date', '')}. "
            f"Notes: {record.get('notes', '')}"
        )

        metadata = {
            "source": "admin_form",
            "type": "reminder",
            "title": record.get("title", ""),
            "date": record.get("date", ""),
            "record_id": record.get("id", "")
        }

    else:
        content = str(record)
        metadata = {
            "source": "admin_form",
            "type": memory_type,
            "record_id": record.get("id", "")
        }

    return Document(
        page_content=content,
        metadata=metadata
    )


def upsert_memory(vector_store, memory_type, record):
    record_id = record.get("id")

    if not record_id:
        return

    item_id = f"{memory_type}_{record_id}"

    try:
        vector_store.delete(ids=[item_id])
    except Exception:
        pass

    vector_store.add_documents(
        [memory_to_document(memory_type, record)],
        ids=[item_id]
    )


def delete_memory(vector_store, memory_type, record_id):
    try:
        vector_store.delete(ids=[f"{memory_type}_{record_id}"])
    except Exception:
        pass


def clean_text(text):
    text = str(text).replace("\r", "\n")
    text = re.sub(r"-+\s*PAGE\s*\d+\s*-+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[━─═―_]{3,}", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_uploaded_document(path):
    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":
        return PyPDFLoader(path).load()

    if extension == ".txt":
        return TextLoader(path, encoding="utf-8").load()

    raise ValueError("Only PDF and TXT files are supported.")


def ingest_document(vector_store, path, filename, document_id):
    raw_docs = load_uploaded_document(path)

    full_text = clean_text(
        "\n".join(
            document.page_content
            for document in raw_docs
            if document.page_content
        )
    )

    if not full_text:
        return 0

    base_doc = Document(
        page_content=full_text,
        metadata={
            "source": "uploaded_document",
            "filename": filename,
            "document_id": document_id,
            "type": "caregiver_document"
        }
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents([base_doc])
    ids = [f"document_{document_id}_chunk_{index}" for index in range(len(chunks))]

    if chunks:
        vector_store.add_documents(chunks, ids=ids)

    return len(chunks)


def delete_document(vector_store, document_id):
    try:
        existing = vector_store.get()
        ids = existing.get("ids", [])
        metadatas = existing.get("metadatas", [])

        delete_ids = [
            item_id
            for item_id, metadata in zip(ids, metadatas)
            if metadata and metadata.get("document_id") == document_id
        ]

        if delete_ids:
            vector_store.delete(ids=delete_ids)

    except Exception:
        pass


def rebuild_index(vector_store, data):
    try:
        existing = vector_store.get()
        ids = existing.get("ids", [])

        if ids:
            vector_store.delete(ids=ids)

    except Exception:
        pass

    docs = []
    ids = []

    for memory_type in ["people", "objects", "routines", "reminders"]:
        for record in data.get(memory_type, []):
            docs.append(memory_to_document(memory_type, record))
            ids.append(f"{memory_type}_{record.get('id')}")

    if docs:
        vector_store.add_documents(docs, ids=ids)

    for record in data.get("documents", []):
        path = record.get("path", "")

        if path and os.path.exists(path):
            ingest_document(
                vector_store,
                path,
                record.get("filename", ""),
                record.get("id", "")
            )


# =========================================================
# UI HELPERS
# =========================================================

def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class='admin-title'>Caregiver Memory Admin</div>
            <div class='admin-subtitle'>
                Add trusted people, objects, routines, reminders, and caregiver documents.
                Everything saved here becomes part of the memory assistant's trusted knowledge base.
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


def render_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{int(value)}</div>
            <div class="metric-label">{escape_html(label)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_card(memory_type, record):
    label = memory_type[:-1] if memory_type.endswith("s") else memory_type

    if memory_type == "people":
        name = record.get("name", "Unnamed person")
        meta = record.get("relationship", "")
        notes = record.get("notes", "")

    elif memory_type == "objects":
        name = record.get("name", "Unnamed object")
        meta = "Object"
        notes = record.get("description", "")

    elif memory_type == "routines":
        name = record.get("title", "Untitled routine")
        meta = f"Time: {record.get('time', '')}"
        notes = record.get("steps", "")

    elif memory_type == "reminders":
        name = record.get("title", "Untitled reminder")
        meta = f"Date: {record.get('date', '')}"
        notes = record.get("notes", "")

    elif memory_type == "documents":
        name = record.get("filename", "Uploaded document")
        meta = f"Chunks indexed: {record.get('chunks', 0)}"
        notes = record.get("path", "")

    else:
        name = "Memory"
        meta = ""
        notes = str(record)

    st.markdown(
        f"""
        <div class="memory-card">
            <div class="memory-type">{escape_html(label)}</div>
            <div class="memory-name">{escape_html(name)}</div>
            <div class="memory-meta">{escape_html(meta)}</div>
            <div class="memory-notes">{escape_html(notes)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FORM SECTIONS
# =========================================================

def add_person(data, vector_store):
    st.subheader("Add Person")

    with st.form("person_form", clear_on_submit=True):
        name = st.text_input("Name", placeholder="Example: Sarah Ahmed")
        relationship = st.text_input("Relationship", placeholder="Example: Daughter")
        notes = st.text_area(
            "Memory Notes",
            placeholder="Example: Sarah visits every Saturday and likes gardening."
        )
        photo = st.file_uploader(
            "Photo (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            key="person_photo"
        )
        submitted = st.form_submit_button("Save Person")

    if submitted:
        if not name.strip():
            st.warning("Please enter a name.")
            return

        record = {
            "id": uuid.uuid4().hex,
            "name": name.strip(),
            "relationship": relationship.strip(),
            "notes": notes.strip(),
            "photo_path": save_photo(photo, "person")
        }

        data["people"].append(record)
        save_store(data)
        upsert_memory(vector_store, "people", record)
        st.success("Person saved and indexed.")


def add_object(data, vector_store):
    st.subheader("Add Object")

    with st.form("object_form", clear_on_submit=True):
        name = st.text_input("Object Name", placeholder="Example: Blue Mug")
        description = st.text_area(
            "Description",
            placeholder="Example: This is the blue mug used for morning tea."
        )
        photo = st.file_uploader(
            "Photo (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            key="object_photo"
        )
        submitted = st.form_submit_button("Save Object")

    if submitted:
        if not name.strip():
            st.warning("Please enter an object name.")
            return

        record = {
            "id": uuid.uuid4().hex,
            "name": name.strip(),
            "description": description.strip(),
            "photo_path": save_photo(photo, "object")
        }

        data["objects"].append(record)
        save_store(data)
        upsert_memory(vector_store, "objects", record)
        st.success("Object saved and indexed.")


def add_routine(data, vector_store):
    st.subheader("Add Routine")

    with st.form("routine_form", clear_on_submit=True):
        title = st.text_input("Routine Title", placeholder="Example: Morning Medicine")
        routine_time = st.text_input("Time", placeholder="Example: 08:00 AM")
        steps = st.text_area(
            "Steps / Notes",
            placeholder="Example:\n1. Take blood pressure medicine.\n2. Drink a glass of water."
        )
        submitted = st.form_submit_button("Save Routine")

    if submitted:
        if not title.strip():
            st.warning("Please enter a routine title.")
            return

        record = {
            "id": uuid.uuid4().hex,
            "title": title.strip(),
            "time": routine_time.strip(),
            "steps": steps.strip()
        }

        data["routines"].append(record)
        save_store(data)
        upsert_memory(vector_store, "routines", record)
        st.success("Routine saved and indexed.")


def add_reminder(data, vector_store):
    st.subheader("Add Reminder")

    with st.form("reminder_form", clear_on_submit=True):
        title = st.text_input("Reminder Title", placeholder="Example: Doctor Appointment")
        reminder_date = st.date_input("Date", value=date.today())
        notes = st.text_area(
            "Notes",
            placeholder="Example: Appointment with Dr. Khan at City Hospital."
        )
        submitted = st.form_submit_button("Save Reminder")

    if submitted:
        if not title.strip():
            st.warning("Please enter a reminder title.")
            return

        record = {
            "id": uuid.uuid4().hex,
            "title": title.strip(),
            "date": str(reminder_date),
            "notes": notes.strip()
        }

        data["reminders"].append(record)
        save_store(data)
        upsert_memory(vector_store, "reminders", record)
        st.success("Reminder saved and indexed.")


def upload_documents(data, vector_store):
    st.subheader("Upload Caregiver Documents")

    st.markdown(
        "Upload TXT or PDF files such as family notes, daily routines, medication schedules, or object descriptions."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF/TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Save and Index Uploaded Documents"):
        if not uploaded_files:
            st.warning("Please choose at least one PDF or TXT file.")
            return

        count = 0

        for uploaded_file in uploaded_files:
            document_id = uuid.uuid4().hex

            try:
                path, filename = save_document_file(uploaded_file)
                chunks = ingest_document(vector_store, path, filename, document_id)

                data["documents"].append({
                    "id": document_id,
                    "filename": filename,
                    "path": path,
                    "chunks": chunks
                })

                count += 1

            except Exception as error:
                st.error(f"Could not ingest {uploaded_file.name}: {error}")

        save_store(data)

        if count:
            st.success(f"{count} document(s) uploaded and indexed.")


def view_memories(data, vector_store):
    st.subheader("Stored Memories")

    labels = {
        "people": "People",
        "objects": "Objects",
        "routines": "Routines",
        "reminders": "Reminders",
        "documents": "Uploaded Documents"
    }

    for memory_type, label in labels.items():
        st.markdown(f"### {label}")

        records = data.get(memory_type, [])

        if not records:
            st.info(f"No {label.lower()} saved yet.")
            continue

        for record in records:
            col1, col2 = st.columns([5, 1])

            with col1:
                render_card(memory_type, record)

            with col2:
                if st.button("Delete", key=f"delete_{memory_type}_{record.get('id')}"):
                    data[memory_type] = [
                        item
                        for item in data[memory_type]
                        if item.get("id") != record.get("id")
                    ]

                    save_store(data)

                    if memory_type == "documents":
                        delete_document(vector_store, record.get("id"))
                    else:
                        delete_memory(vector_store, memory_type, record.get("id"))

                    st.success("Deleted successfully.")
                    st.rerun()


# =========================================================
# MAIN
# =========================================================

vector_store = get_vector_store()
data = load_store()

total_sources = (
    len(data["people"])
    + len(data["objects"])
    + len(data["routines"])
    + len(data["reminders"])
    + len(data["documents"])
)

with st.sidebar:
    render_sidebar_logo()

    st.markdown(
        """
        <div class="sidebar-title">Memory Admin</div>
        <div class="sidebar-subtitle">
            Manage trusted memories for the assistant.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='sidebar-section-label'>Workspace</div>", unsafe_allow_html=True)

    page = st.radio(
        "Section",
        [
            "Dashboard",
            "Add Person",
            "Add Object",
            "Add Routine",
            "Add Reminder",
            "Upload Documents",
            "View Memories"
        ],
        label_visibility="collapsed"
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

    if st.button("Rebuild Memory Index", use_container_width=True):
        rebuild_index(vector_store, data)
        st.success("Memory index rebuilt.")

    st.markdown(
        """
        <div class="sidebar-help">
            Tip: add memories through forms for structured data, or upload PDF/TXT files for existing caregiver notes.
        </div>
        """,
        unsafe_allow_html=True
    )

render_header()

if page == "Dashboard":
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        render_metric_card("People", len(data["people"]))
    with col2:
        render_metric_card("Objects", len(data["objects"]))
    with col3:
        render_metric_card("Routines", len(data["routines"]))
    with col4:
        render_metric_card("Reminders", len(data["reminders"]))
    with col5:
        render_metric_card("Documents", len(data["documents"]))

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="demo-flow">
            <h3>Suggested Demo Flow</h3>
            <ol>
                <li>Add a familiar person such as a daughter, son, spouse, or caregiver.</li>
                <li>Add a daily routine such as morning medicine.</li>
                <li>Add a meaningful object such as a favorite mug or walking stick.</li>
                <li>Upload a family notes or routine document.</li>
                <li>Open the patient app and ask: <strong>"Who is Sarah?"</strong> or <strong>"What medicine do I take in the morning?"</strong></li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "Add Person":
    add_person(data, vector_store)

elif page == "Add Object":
    add_object(data, vector_store)

elif page == "Add Routine":
    add_routine(data, vector_store)

elif page == "Add Reminder":
    add_reminder(data, vector_store)

elif page == "Upload Documents":
    upload_documents(data, vector_store)

elif page == "View Memories":
    view_memories(data, vector_store)
