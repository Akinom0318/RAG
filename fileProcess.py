import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


DATA_PATH = "pdfs"
CHROMA_PATH = "chroma"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    print(chunks)
    add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len,
        is_separator_regex = False,
    )

    return splitter.split_documents(documents)

def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embeddings

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = get_embedding_function()
    )
    chunk_ids = create_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunk_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks to the DB")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids = new_chunks_ids)
        db.persist()
    else:
        print("No new chunks to add")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def create_chunk_ids(chunks: list[Document]):
    # create id like: Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    return chunks

if __name__ == "__main__":
    main()