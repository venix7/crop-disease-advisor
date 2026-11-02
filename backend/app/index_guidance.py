import json
from pathlib import Path
from uuid import uuid4

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import delete, text

from .database import Base, DocumentChunk, SessionLocal, engine


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "config" / "project.json").read_text(encoding="utf-8")
)
LOCAL_GUIDANCE_DIR = ROOT / "data" / "guidance"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_article(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("main") or soup.find("article")

    if content is None:
        raise ValueError(f"No main/article element found: {url}")

    for element in content.find_all(
        ["script", "style", "nav", "footer", "form", "aside"]
    ):
        element.decompose()

    lines = [
        line.strip()
        for line in content.get_text("\n", strip=True).splitlines()
    ]
    return "\n".join(line for line in lines if line)


def fetch_page(source: dict) -> str:
    local_file = LOCAL_GUIDANCE_DIR / f"{source['class_id']}.txt"

    if local_file.is_file():
        print(f"Using local text: {local_file}")
        cleaned = local_file.read_text(encoding="utf-8")
    else:
        try:
            response = requests.get(
                source["url"],
                headers={
                    "User-Agent": (
                        "CropDiseaseAdvisor/1.0 "
                        "(educational project; document indexing)"
                    )
                },
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not fetch {source['url']}. Open that URL in your "
                f"browser, copy the article text, and save it as "
                f"data/guidance/{source['class_id']}.txt. Then rerun "
                f"the indexing command."
            ) from exc

        cleaned = extract_article(response.text, source["url"])

    cleaned = cleaned.strip()
    if len(cleaned) < 300:
        raise ValueError(
            f"Guidance text for {source['class_id']} is under "
            f"300 characters. Save the full article text."
        )

    return cleaned


def main():
    LOCAL_GUIDANCE_DIR.mkdir(parents=True, exist_ok=True)

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
    )
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    for source in CONFIG["sources"]:
        print(f"Indexing: {source['title']}")

        page_text = fetch_page(source)
        passages = splitter.split_text(page_text)

        vectors = embedding_model.encode(
            passages,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        with SessionLocal.begin() as session:
            session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.class_id == source["class_id"],
                    DocumentChunk.source_url == source["url"],
                )
            )

            for index, (passage, vector) in enumerate(
                zip(passages, vectors)
            ):
                session.add(
                    DocumentChunk(
                        id=str(uuid4()),
                        class_id=source["class_id"],
                        source_title=source["title"],
                        source_url=source["url"],
                        chunk_index=index,
                        content=passage,
                        embedding=vector.tolist(),
                    )
                )

        print(f"Stored {len(passages)} passages")

    print("Indexing complete.")


if __name__ == "__main__":
    main()