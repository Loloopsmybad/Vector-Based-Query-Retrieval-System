"""
search.py — Semantic search over your indexed books.

Usage:
    python search.py                        # interactive mode
    python search.py "your query here"      # single query mode
    python search.py "query" --top 10       # return top 10 results
"""

import sys
import argparse
import lancedb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_DIR  = Path("index")
TABLE_NAME = "books"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
# ─────────────────────────────────────────────────────────────────────────────

def format_result(rank: int, result: dict) -> str:
    """Pretty-print a single search result."""
    score = 1 - result.get("_distance", 0)   # LanceDB returns distance; convert to similarity
    snippet = result["text"][:300].replace("\n", " ")
    if len(result["text"]) > 300:
        snippet += "..."

    lines = [
        f"\n{'─'*60}",
        f"  #{rank}  Score: {score:.3f}  |  {result['file_name']}  |  Page {result['page']}",
        f"{'─'*60}",
        f"  {snippet}",
    ]
    return "\n".join(lines)


def search(query: str, table, model: SentenceTransformer, top_k: int) -> list[dict]:
    """Embed query and retrieve top_k nearest chunks."""
    query_vec = model.encode(query).tolist()
    results = (
        table.search(query_vec)
             .limit(top_k)
             .to_list()
    )
    return results


def show_index_stats(table):
    """Print a summary of what's currently indexed."""
    total = table.count_rows()
    if total == 0:
        print("⚠  Index is empty. Run 'python ingest.py' first.")
        return False

    # Get unique books
    all_rows = table.search(None).limit(total).to_list()
    books = sorted(set(r["file_name"] for r in all_rows))

    print(f"\n📚 Index contains {total} chunks across {len(books)} book(s):")
    for b in books:
        count = sum(1 for r in all_rows if r["file_name"] == b)
        print(f"   • {b}  ({count} chunks)")
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description="Semantic book search")
    parser.add_argument("query", nargs="?", help="Search query (omit for interactive mode)")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_K, help="Number of results")
    args = parser.parse_args()

    # Connect
    if not INDEX_DIR.exists():
        print("⚠  No index found. Run 'python ingest.py' first.")
        sys.exit(1)

    db = lancedb.connect(str(INDEX_DIR))
    if TABLE_NAME not in db.table_names():
        print("⚠  No index table found. Run 'python ingest.py' first.")
        sys.exit(1)

    table = db.open_table(TABLE_NAME)

    # Load model
    print(f"🔧 Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"   ✓  Ready\n")

    if not show_index_stats(table):
        sys.exit(1)

    # Single query mode
    if args.query:
        results = search(args.query, table, model, args.top)
        print(f"🔍 Query: \"{args.query}\"  |  Top {args.top} results\n")
        for i, r in enumerate(results, 1):
            print(format_result(i, r))
        print()
        return

    # Interactive mode
    print("💬 Interactive search mode. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("🔍 Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        results = search(query, table, model, args.top)

        if not results:
            print("   No results found.\n")
            continue

        for i, r in enumerate(results, 1):
            print(format_result(i, r))
        print()


if __name__ == "__main__":
    main()
