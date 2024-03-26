import os
import chromadb
from langchain.vectorstores import Chroma
client_settings = chromadb.config.Settings(
    is_persistent=True,
    persist_directory='DB_DIR',
    anonymized_telemetry=False,
)