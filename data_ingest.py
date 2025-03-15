from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os
import logging
import pymupdf # imports the pymupdf library

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def pdf_upload():
    try:
        # Get raw text
        raw_text = get_pdf_text()
        # Text chunks
        text_chunks = get_text_chunks(raw_text)
        # Embedding
        store_vector(text_chunks, "documents")

    except Exception as e:
        logging.error("Fail to upload pdf", e)


"""
Functions for storing pdf file to chromadb
"""
def get_pdf_text():
    text = ""
    pdfs = ["SOFI-2023.pdf", "SOFI-2024.pdf"]

    for pdf in pdfs:
        doc = pymupdf.open(pdf)
        for page in doc: # iterate the document pages
            text += page.get_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    chunks = text_splitter.split_text(text)
    return chunks


def store_vector(text_chunks, collection):
    api_key = OPENAI_API_KEY
    embedding = OpenAIEmbeddings(api_key=api_key)

    vector_db = Chroma.from_texts(
        text_chunks,
        embedding=embedding,
        persist_directory="./data",
        collection_name="documents",
    )
    return vector_db


# populate
pdf_upload()
