"""
watch.py — Watches the books/ folder and auto-indexes any new PDF dropped in.

Usage:
    python watch.py

Keep this running in a terminal. Whenever you drop a new PDF into books/,
it will automatically be extracted, embedded, and added to the index.
"""

import time
import lancedb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import the pipeline from ingest.py
from ingest import ingest_pdf, INDEX_DIR, TABLE_NAME, MODEL_NAME, CHUNK_SIZE

# ── Config ────────────────────────────────────────────────────────────────────
BOOKS_DIR   = Path("books")
POLL_DELAY  = 2   # seconds to wait after a file event (avoid partial writes)
# ─────────────────────────────────────────────────────────────────────────────


class PDFHandler(FileSystemEventHandler):
    def __init__(self, table, model):
        self.table = table
        self.model = model
        self._pending = set()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".pdf"):
            self._pending.add(event.src_path)

    def on_moved(self, event):
        """Handles files moved/renamed into the watch folder."""
        if not event.is_directory and event.dest_path.endswith(".pdf"):
            self._pending.add(event.dest_path)

    def process_pending(self):
        """Process any queued PDFs (called from main loop with a delay)."""
        for path_str in list(self._pending):
            path = Path(path_str)
            if path.exists() and path.stat().st_size > 0:
                print(f"\n👁  Detected new file: {path.name}")
                try:
                    ingest_pdf(path, self.table, self.model)
                    print(f"   ✅ '{path.name}' indexed successfully.")
                    print(f"   📊 Index now has {self.table.count_rows()} chunks total.\n")
                except Exception as e:
                    print(f"   ❌ Failed to index '{path.name}': {e}\n")
                self._pending.discard(path_str)


def main():
    BOOKS_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)

    print(f"🔧 Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"   ✓  Model ready\n")

    db = lancedb.connect(str(INDEX_DIR))
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
    else:
        table = db.create_table(TABLE_NAME, data=schema_record)
        table.delete("file_name = ''")

    handler = PDFHandler(table, model)
    observer = Observer()
    observer.schedule(handler, str(BOOKS_DIR), recursive=False)
    observer.start()

    print(f"👁  Watching '{BOOKS_DIR}/' for new PDFs...")
    print(f"   Drop any PDF into that folder and it will be indexed automatically.")
    print(f"   Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(POLL_DELAY)
            handler.process_pending()
    except KeyboardInterrupt:
        print("\n🛑 Stopping watcher...")
        observer.stop()

    observer.join()
    print("Done.")


if __name__ == "__main__":
    main()
