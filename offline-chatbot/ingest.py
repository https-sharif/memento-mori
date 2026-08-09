import os
import re
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = str(Path(__file__).resolve().parent)

DATA_DIR = os.path.join(BASE_DIR, "memory_data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_memory")
UPLOAD_DIR = os.path.join(BASE_DIR, "caregiver_documents")

for folder in [DATA_DIR, CHROMA_DIR, UPLOAD_DIR]:
    os.makedirs(folder, exist_ok=True)


def clean_text(text):
    text = str(text).replace("\r", "\n")
    text = re.sub(r"-+\s*PAGE\s*\d+\s*-+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[━─═―_]{3,}", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_file(path):
    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":
        return PyPDFLoader(path).load()

    if extension == ".txt":
        return TextLoader(path, encoding="utf-8").load()

    return []


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    supported_extensions = {".pdf", ".txt"}

    files = [
        filename
        for filename in os.listdir(UPLOAD_DIR)
        if os.path.splitext(filename)[1].lower() in supported_extensions
    ]

    if not files:
        print("No PDF/TXT files found in caregiver_documents.")
        return

    total_chunks = 0

    for filename in files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        document_id = uuid.uuid4().hex

        print(f"Ingesting: {filename}")

        raw_docs = load_file(file_path)

        full_text = clean_text(
            "\n".join(
                document.page_content
                for document in raw_docs
                if document.page_content
            )
        )

        if not full_text:
            print(f"Skipped empty file: {filename}")
            continue

        base_document = Document(
            page_content=full_text,
            metadata={
                "source": "batch_ingest",
                "filename": filename,
                "document_id": document_id,
                "type": "caregiver_document"
            }
        )

        chunks = splitter.split_documents([base_document])

        ids = [
            f"batch_document_{document_id}_chunk_{index}"
            for index in range(len(chunks))
        ]

        if chunks:
            vector_store.add_documents(chunks, ids=ids)
            total_chunks += len(chunks)

        print(f"Indexed {len(chunks)} chunk(s).")

    print(f"Done. Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    main()
