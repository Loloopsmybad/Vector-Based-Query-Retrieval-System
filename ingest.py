"""
ingest.py — Extract, chunk, embed and index PDFs into LanceDB.

Usage:
    python ingest.py                   # indexes all PDFs in ./books/
    python ingest.py path/to/file.pdf  # indexes a single PDF
"""

import os
import sys
import hashlib
import lancedb
import pdfplumber
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
BOOKS_DIR   = Path("books")
INDEX_DIR   = Path("index")
TABLE_NAME  = "books"
MODEL_NAME  = "all-MiniLM-L6-v2"   # ~80MB, fast on CPU
CHUNK_SIZE  = 400                   # words per chunk
CHUNK_OVERLAP = 80                  # overlap between chunks (for context continuity)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_by_page(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF. Returns list of {page, text}."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    return pages


def chunk_pages(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """
    Combine all page text into word-level chunks with overlap.
    Each chunk carries the page number where it starts.
    """
    chunks = []
    # Build a flat word list annotated with page numbers
    words_with_pages = []
    for p in pages:
        for word in p["text"].split():
            words_with_pages.append((word, p["page"]))

    step = chunk_size - overlap
    i = 0
    while i < len(words_with_pages):
        window = words_with_pages[i : i + chunk_size]
        chunk_text = " ".join(w for w, _ in window)
        start_page = window[0][1]
        chunks.append({"page": start_page, "text": chunk_text})
        i += step

    return chunks


def file_hash(path: Path) -> str:
    """MD5 hash of file contents — used to detect already-indexed files."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def already_indexed(table, path: Path) -> bool:
    """Return True if this file hash already exists in the index."""
    fhash = file_hash(path)
    try:
        results = table.search(None).where(f"file_hash = '{fhash}'").limit(1).to_list()
        return len(results) > 0
    except Exception:
        return False


def ingest_pdf(pdf_path: Path, table, model: SentenceTransformer):
    """Full pipeline for one PDF: extract → chunk → embed → store."""
    print(f"\n📖 Processing: {pdf_path.name}")

    # Skip if already indexed
    fhash = file_hash(pdf_path)
    try:
        existing = table.search(None).where(f"file_hash = '{fhash}'").limit(1).to_list()
        if existing:
            print(f"   ⏭  Already indexed, skipping.")
            return
    except Exception:
        pass

    # Extract
    pages = extract_text_by_page(pdf_path)
    if not pages:
        print(f"   ⚠  No text found (scanned PDF? Try OCR first).")
        return
    print(f"   ✓  Extracted text from {len(pages)} pages")

    # Chunk
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"   ✓  Created {len(chunks)} chunks")

    # Embed
    texts = [c["text"] for c in chunks]
    print(f"   ⏳ Embedding {len(texts)} chunks (this takes a moment)...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"   ✓  Embedding done")

    # Build records
    records = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        records.append({
            "vector":    vec.tolist(),
            "text":      chunk["text"],
            "file_name": pdf_path.name,
            "file_path": str(pdf_path.resolve()),
            "file_hash": fhash,
            "page":      chunk["page"],
            "chunk_id":  i,
        })

    # Store
    table.add(records)
    print(f"   ✓  Indexed {len(records)} chunks from '{pdf_path.name}'")


def main():
    BOOKS_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)

    # Load model once
    print(f"🔧 Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"   ✓  Model loaded (vector dim: {model.get_sentence_embedding_dimension()})")

    # Connect to LanceDB
    db = lancedb.connect(str(INDEX_DIR))

    # Create or open table
    dim = model.get_sentence_embedding_dimension()
    schema_record = [{
        "vector":    [0.0] * dim,
        "text":      "",
        "file_name": "",
        "file_path": "",
        "file_hash": "",
        "page":      0,
        "chunk_id":  0,
    }]

    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        print(f"📂 Opened existing index with {table.count_rows()} chunks")
    else:
        table = db.create_table(TABLE_NAME, data=schema_record)
        # Remove the dummy seed record
        table.delete("file_name = ''")
        print(f"📂 Created new index")

    # Determine which PDFs to process
    if len(sys.argv) > 1:
        pdfs = [Path(sys.argv[1])]
    else:
        pdfs = sorted(BOOKS_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"\n⚠  No PDFs found in '{BOOKS_DIR}/'. Drop some PDFs there and re-run.")
        return

    print(f"\n📚 Found {len(pdfs)} PDF(s) to process")
    for pdf_path in pdfs:
        ingest_pdf(pdf_path, table, model)

    print(f"\n✅ Done! Index now contains {table.count_rows()} total chunks.")
    print(f"   Run 'python search.py' to start searching.")


if __name__ == "__main__":
    main()
