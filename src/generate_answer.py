import os
import time
import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "paragraph_chunks"

MODEL_NAME = "gemini-3.8-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 15
# ============================================================
# GEMINI
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found.\n"
        "Please set your Gemini API key."
    )

client = genai.Client(api_key=api_key)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_collection(
    COLLECTION_NAME
)

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(question, top_k=TOP_K):
    
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results["documents"][0]
    logger.info(f"Retrieved Chunks: {len(documents)}")
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas, distances
# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents, metadatas):

    parts = []

    for i, (doc, meta) in enumerate(
        zip(documents, metadatas),
        start=1
    ):

        parts.append(
            f"""
            SOURCE {i}

            Document: {meta.get("document")}

            Page: {meta.get("page")}

            Passage:
            {doc}
            """
        )

    return "\n".join(parts)

# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    documents, metadatas, distances = retrieve_chunks(
        question,
        TOP_K
    )

    logger.info("\n========== RETRIEVED DOCUMENTS ==========")

    for i, (doc, meta, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):
        logger.info(f"\nChunk {i}")
        logger.info(f"Document : {meta.get('document')}")
        logger.info(f"Page     : {meta.get('page')}")
        logger.info(f"Distance : {distance:.4f}")
        logger.info(doc)



    logger.info("\n=========================================")

    if len(documents) == 0:
        return "I don't know based on the provided documents."

    context = build_context(
        documents,
        metadatas
    )
    logger.info("=" * 60)
    logger.info(f"Context Size: {len(context)}")
    logger.info("=" * 60)

    prompt = f"""
You are an intelligent Government Report Question Answering Assistant.

You MUST answer using ONLY the retrieved evidence.

Rules:

1. Read ALL retrieved passages carefully.

2. If the answer appears in ANY passage,
   answer it.

3. Combine multiple passages when needed.

4. Never use outside knowledge.

5. Never invent facts.

6. Quote important facts from the evidence.

7. Mention the document name and page number.

8. Only answer

I don't know based on the provided documents.

when absolutely no retrieved passage contains the answer.

Retrieved Evidence:

{context}

Question:

{question}

Answer:
"""
    response = None

    for attempt in range(3):

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                ),
            )

            if getattr(response, "text", None):
                return response.text.strip()

            break

        except Exception as e:

            error = str(e)

            logger.error(f"Attempt {attempt+1} failed")
            logger.error(error)

            if "429" in error or "RESOURCE_EXHAUSTED" in error:
                return (
                    "Gemini API daily quota exceeded.\n"
                    "Please wait until tomorrow or use another API key."
                )

            time.sleep(2 ** attempt)

    # ---------- OUTSIDE THE LOOP ----------

    if response is None:
        return "Gemini API temporarily unavailable."

    if getattr(response, "text", None):
        return response.text.strip()

    return "I don't know based on the provided documents."

# ============================================================
# RAG RESPONSE
# ============================================================

def get_rag_response(question):

    documents, metadatas, distances = retrieve_chunks(
        question,
        TOP_K
    )

    answer = generate_answer(question)

    return {
        "question": question,
        "answer": answer,
        "contexts": documents,
        "metadata": metadatas,
    }

# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info("Government Report Intelligence - Document Q&A")
    logger.info("=" * 70)

    logger.info(f"\nVector Database : {CHROMA_PATH}")
    logger.info(f"Collection      : {COLLECTION_NAME}")
    logger.info(f"Model           : {MODEL_NAME}")
    logger.info(f"Top-K           : {TOP_K}")

    logger.info("\nSystem Ready!")

    while True:

        question = input(
            "\nEnter your question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer = generate_answer(question)

        logger.info("\n" + "=" * 70)
        logger.info("ANSWER")
        logger.info("=" * 70)
        logger.info(answer)
        logger.info("=" * 70)

if __name__ == "__main__":
    main()