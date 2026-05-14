# Book Search — Vector-Based Semantic Search

Search your PDFs by meaning, not just keywords.

## Setup

```bash
pip install -r requirements.txt
mkdir books
```

## Usage

### Step 1 — Drop PDFs into the books/ folder
```
book-search/
└── books/
    ├── my_book.pdf
    └── another.pdf
```

### Step 2 — Index them
```bash
python ingest.py
```
This extracts text, chunks it, embeds each chunk using sentence-transformers,
and stores everything in LanceDB under the index/ folder.

To index a single file:
```bash
python ingest.py books/my_book.pdf
```

### Step 3 — Search
```bash
python search.py                         # interactive mode
python search.py "what is dark matter"   # single query
python search.py "machine learning" --top 10
```

### Optional — Auto-watch for new PDFs
```bash
python watch.py
```
Keeps running in the background. Any PDF you drop into books/ gets
automatically indexed without re-running ingest.py.

## How it works

```
PDF file
  └─► pdfplumber extracts text page by page
        └─► text is split into overlapping 400-word chunks
              └─► sentence-transformers embeds each chunk → 384-dim vector
                    └─► vectors + metadata stored in LanceDB (HNSW index)
                          └─► search: embed query → find nearest vectors
                                └─► return ranked passages + page numbers
```

## File structure after running

```
book-search/
├── books/          ← your PDFs go here
├── index/          ← LanceDB vector store (auto-created)
├── ingest.py       ← indexing pipeline
├── search.py       ← search CLI
├── watch.py        ← auto-watcher
└── requirements.txt
```

## Notes

- First run downloads the embedding model (~80MB), cached after that
- A 300-page book takes ~2–3 minutes to index on CPU
- Search is always milliseconds regardless of index size
- Already-indexed files are skipped automatically (hash-based deduplication)
- Scanned PDFs (images of pages) won't work — text must be selectable
