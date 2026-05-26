print("Script started")

import os                        # built-in Python — lets us work with files and folders
import glob                      # built-in Python — finds files matching a pattern (like *.md)
from dotenv import load_dotenv   # reads your .env file and loads your API keys

from langchain_text_splitters import RecursiveCharacterTextSplitter
# ↑ This is the tool that splits long text into smaller chunks

import chromadb
# ↑ This is the vector database that stores your chunks on disk

from chromadb.utils import embedding_functions
# ↑ This tells ChromaDB HOW to convert text into numbers (embeddings)


# --- LOAD ENVIRONMENT VARIABLES ---
# This reads your .env file and makes the values available in the code
load_dotenv()

# Read the paths and model names from your .env file
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# ↑ os.getenv reads a value from .env. The second argument is a default value
#   if the key is not found in .env

DOCS_PATH = "fastapi/docs/en/docs"
# ↑ This is the folder where the FastAPI .md files live
#   Change this if your folder structure is different

CHROMA_DB_PATH = "./chroma_db"
# ↑ This is where ChromaDB will save the database on your computer
#   A folder called "chroma_db" will be created in your task1 folder

COLLECTION_NAME = "fastapi_docs"
# ↑ Think of a collection like a table in a database — it holds all your chunks


# --- STEP 1: FIND ALL .md FILES ---
def find_markdown_files(docs_path):
    """
    Finds every .md file in the given folder and all its subfolders.
    Returns a list of file paths.
    """
    # glob.glob with ** and recursive=True searches ALL subfolders
    pattern = os.path.join(docs_path, "**", "*.md")
    files = glob.glob(pattern, recursive=True)

    print(f"Found {len(files)} markdown files in {docs_path}")
    return files


# --- STEP 2: READ A FILE AND EXTRACT TEXT ---
def read_markdown_file(file_path):
    """
    Opens a single .md file and returns its text content.
    Also returns the filename so we can use it as a citation later.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        # If a file can't be read (encoding issue etc), skip it and print a warning
        print(f"  WARNING: Could not read {file_path} — {e}")
        return None


# --- STEP 3: SPLIT TEXT INTO CHUNKS ---
def create_chunks(text, file_path):
    """
    Splits a long piece of text into smaller chunks.

    Why do we chunk?
    Because AI models can't meaningfully compare a whole page to a short question.
    Small chunks of ~500 characters work much better.

    chunk_size=500      — each chunk is at most 500 characters
    chunk_overlap=50    — the last 50 characters of one chunk are repeated at the
                          start of the next chunk, so meaning doesn't get cut off
                          at the boundary
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
        # ↑ It tries to split at double newlines first (paragraphs),
        #   then single newlines, then spaces, then characters.
        #   This keeps sentences and paragraphs together when possible.
    )

    chunks = splitter.split_text(text)

    # For each chunk, we also store metadata (where it came from)
    # This is how citations work — we know which file each answer came from
    metadatas = []
    for i, chunk in enumerate(chunks):
        metadatas.append({
            "source": file_path,           # full file path
            "filename": os.path.basename(file_path),  # just the filename e.g. "tutorial.md"
            "chunk_index": i               # which chunk number within this file
        })

    return chunks, metadatas


# --- STEP 4: SET UP CHROMADB ---
def setup_chromadb():
    """
    Creates (or connects to) the ChromaDB database on your computer.
    If the database already exists, it deletes the old collection and
    starts fresh — this makes the script idempotent (safe to run twice).
    """
    print(f"Setting up ChromaDB at {CHROMA_DB_PATH}...")

    # PersistentClient saves data to disk (in the chroma_db folder)
    # Without this, data would disappear when you close the terminal
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Set up the embedding function using HuggingFace sentence-transformers
    # This is what converts text → numbers
    # The model downloads automatically the first time (may take 1-2 mins)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Delete existing collection if it exists (so running twice is safe)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    # Create a fresh collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
        # ↑ cosine similarity means: find chunks whose MEANING is closest
        #   to the question, not just chunks that share exact words
    )

    print(f"  Created collection '{COLLECTION_NAME}'")
    return collection


# --- STEP 5: SAVE CHUNKS TO CHROMADB ---
def add_chunks_to_collection(collection, all_chunks, all_metadatas, all_ids):
    """
    Saves all chunks into the ChromaDB collection.
    ChromaDB needs three things for each chunk:
      - documents: the actual text
      - metadatas: where it came from (filename, chunk index)
      - ids: a unique ID string for each chunk
    We save in batches of 100 to avoid memory issues with large corpora.
    """
    batch_size = 100  # save 100 chunks at a time

    for i in range(0, len(all_chunks), batch_size):
        batch_chunks    = all_chunks[i : i + batch_size]
        batch_metadatas = all_metadatas[i : i + batch_size]
        batch_ids       = all_ids[i : i + batch_size]

        collection.add(
            documents=batch_chunks,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    print(f"  Saved {len(all_chunks)} chunks to ChromaDB")


# --- MAIN FUNCTION — ties everything together ---
def main():
    print("=" * 50)
    print("FastAPI Docs Ingestion Starting...")
    print("=" * 50)

    # Step 1: Find all .md files
    files = find_markdown_files(DOCS_PATH)

    if len(files) == 0:
        print("ERROR: No .md files found!")
        print(f"Make sure the FastAPI docs are at: {DOCS_PATH}")
        print("Run: git clone https://github.com/tiangolo/fastapi")
        return

    # Step 2: Set up the database
    collection = setup_chromadb()

    # Step 3: Process every file
    all_chunks    = []  # will hold all text chunks from all files
    all_metadatas = []  # will hold metadata for each chunk
    all_ids       = []  # will hold a unique ID for each chunk

    for file_num, file_path in enumerate(files, start=1):
        filename = os.path.basename(file_path)
        print(f"Processing ({file_num}/{len(files)}): {filename}")

        # Read the file
        text = read_markdown_file(file_path)
        if text is None or text.strip() == "":
            print(f"  Skipping — empty or unreadable")
            continue  # skip this file and move to the next

        # Split into chunks
        chunks, metadatas = create_chunks(text, file_path)

        if len(chunks) == 0:
            print(f"  Skipping — no chunks created")
            continue

        # Create unique IDs for each chunk
        # Format: "filename__chunk_0", "filename__chunk_1" etc
        ids = [f"{file_num}_{filename}__chunk_{i}" for i in range(len(chunks))]

        # Add to our running lists
        all_chunks.extend(chunks)
        all_metadatas.extend(metadatas)
        all_ids.extend(ids)

        print(f"  Created {len(chunks)} chunks")

    # Step 4: Save everything to ChromaDB
    print(f"\nSaving {len(all_chunks)} total chunks to database...")
    add_chunks_to_collection(collection, all_chunks, all_metadatas, all_ids)

    # Done!
    print("\n" + "=" * 50)
    print("INGESTION COMPLETE!")
    print(f"Total files processed : {len(files)}")
    print(f"Total chunks saved    : {len(all_chunks)}")
    print(f"Database saved at     : {CHROMA_DB_PATH}")
    print("=" * 50)
    print("\nYou can now build retrieval.py on top of this database.")


# --- RUN THE SCRIPT ---
# This line means: only run main() if you run this file directly
# (not if another file imports it)
if __name__ == "__main__":
    main()