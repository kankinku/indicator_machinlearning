import argparse
from pathlib import Path

from src.ontology.sa_rag import Document, SpreadingActivationRAG
from src.shared.logger import get_logger

logger = get_logger("tools.sa_rag_demo")


SAMPLE_DOCS = [
    Document(
        doc_id="sample_1",
        text=(
            "Company A relies on Supplier B for rare metals. "
            "Supplier B sources key materials from Commodity C markets. "
            "A sharp increase in Commodity C prices can compress margins for Company A."
        ),
    ),
    Document(
        doc_id="sample_2",
        text=(
            "Supplier B reports rising input costs after Commodity C futures rallied. "
            "Company A has not updated long-term contracts yet."
        ),
    ),
    Document(
        doc_id="sample_3",
        text=(
            "Unrelated sector news: Company D expands into new markets. "
            "No direct link to Company A supply chain."
        ),
    ),
]


def load_documents(paths):
    if not paths:
        return SAMPLE_DOCS

    docs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            logger.warning(f"Missing file: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(doc_id=path.stem, text=text))
    return docs or SAMPLE_DOCS


def main():
    parser = argparse.ArgumentParser(description="SA-RAG demo (ontology zone).")
    parser.add_argument("--query", required=False, default="Why is Company A margin declining?")
    parser.add_argument("--file", nargs="*", help="Text files to index")
    args = parser.parse_args()

    rag = SpreadingActivationRAG()
    docs = load_documents(args.file)
    rag.index_documents(docs)
    result = rag.query(args.query)

    print("=== SA-RAG CONTEXT ===")
    print(result.context)
    print("\n=== ACTIVATED NODES ===")
    for node_id, score in sorted(result.activated_nodes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{node_id}: {score:.3f}")


if __name__ == "__main__":
    main()
