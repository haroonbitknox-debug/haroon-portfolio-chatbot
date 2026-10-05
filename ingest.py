"""Build the vector index from everything in ./data.

Run once locally (and again whenever you edit your content):
    python ingest.py
Then commit the generated ./index folder so the server can load it at startup.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

ROOT = Path(__file__).parent
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

loader = DirectoryLoader(
    str(ROOT / "data"), glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=60,
    separators=["\n## ", "\n\n", "\n", ". ", " "],
)
chunks = splitter.split_documents(docs)

FAISS.from_documents(chunks, GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)).save_local(
    str(ROOT / "index")
)
print(f"Indexed {len(docs)} files into {len(chunks)} chunks -> {ROOT / 'index'}")
