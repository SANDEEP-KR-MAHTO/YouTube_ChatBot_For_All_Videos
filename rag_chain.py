import logging
from typing import Generator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import DEFAULT_GROQ_MODEL, RERANKER_MODEL, TOP_K_RERANK, TOP_K_RETRIEVE
from transcript import format_timestamp
from vectorstore import hybrid_search

logger = logging.getLogger(__name__)

# ── Reranker (lazy singleton) ─────────────────────────────────────────────────

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ── Retrieval + Reranking ─────────────────────────────────────────────────────

def retrieve_and_rerank(
    vectorstore: FAISS,
    bm25: BM25Okapi | None,
    docs_corpus: list[Document] | None,
    query: str,
    k_retrieve: int = TOP_K_RETRIEVE,
    k_rerank: int = TOP_K_RERANK,
) -> list[Document]:
    """Hybrid search → cross-encoder rerank → top-k docs."""
    candidates = hybrid_search(vectorstore, bm25, docs_corpus, query, k=k_retrieve)

    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[:k_rerank]]

    if ranked:
        logger.debug(
            f"Reranked {len(candidates)} → {len(top_docs)} docs. "
            f"Top score: {ranked[0][0]:.3f}"
        )
    return top_docs


# ── Formatting ────────────────────────────────────────────────────────────────

def format_docs_with_timestamps(docs: list[Document]) -> str:
    """Build the context string injected into the prompt, with timestamp labels."""
    parts = []
    for doc in docs:
        start = doc.metadata.get("start")
        end = doc.metadata.get("end")
        if start is not None and end is not None:
            label = f"[{format_timestamp(start)} – {format_timestamp(end)}]"
            parts.append(f"{label}\n{doc.page_content}")
        else:
            parts.append(doc.page_content)
    return "\n\n---\n\n".join(parts)


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions based strictly on "
                "the provided YouTube video transcript excerpts.\n\n"
                "If the answer is not found in the transcript, say "
                "\"I couldn't find that information in the video transcript.\"\n\n"
                "Transcript excerpts:\n{context}",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )


def _build_chain(model: str):
    """Return a LangChain LCEL chain: prompt | llm | str_parser."""
    llm = ChatGroq(model=model)
    return _build_prompt() | llm | StrOutputParser()


def _history_to_messages(chat_history: list[dict]) -> list[tuple[str, str]]:
    return [(turn["role"], turn["content"]) for turn in chat_history[-6:]]


# ── Public API ────────────────────────────────────────────────────────────────

def stream_response(
    context: str,
    query: str,
    chat_history: list[dict],
    model: str = DEFAULT_GROQ_MODEL,
) -> Generator[str, None, None]:
    """
    Yield response tokens one by one (for st.write_stream).
    Call retrieve_and_rerank separately to get source docs.
    """
    chain = _build_chain(model)
    logger.info(f"Streaming answer | model={model}")
    yield from chain.stream(
        {
            "context": context,
            "chat_history": _history_to_messages(chat_history),
            "question": query,
        }
    )


def generate_answer(
    vectorstore: FAISS,
    bm25: BM25Okapi | None,
    docs_corpus: list[Document] | None,
    query: str,
    chat_history: list[dict],
    model: str = DEFAULT_GROQ_MODEL,
) -> tuple[str, list[Document]]:
    """Non-streaming answer. Returns (answer_text, source_docs)."""
    retrieved = retrieve_and_rerank(vectorstore, bm25, docs_corpus, query)
    context = format_docs_with_timestamps(retrieved)
    chain = _build_chain(model)
    logger.info(f"Generating answer | model={model} | chunks={len(retrieved)}")
    answer = chain.invoke(
        {
            "context": context,
            "chat_history": _history_to_messages(chat_history),
            "question": query,
        }
    )
    return answer, retrieved


def generate_summary(
    docs_corpus: list[Document],
    model: str = DEFAULT_GROQ_MODEL,
) -> str:
    """
    Summarise the video in 4-6 bullet points using a spread sample of chunks
    so the summary covers the whole video, not just the beginning.
    """
    n = len(docs_corpus)
    sample_size = min(15, n)
    if n <= sample_size:
        sample = docs_corpus
    else:
        step = n / sample_size
        sample = [docs_corpus[int(i * step)] for i in range(sample_size)]

    context = "\n\n".join(doc.page_content for doc in sample)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Summarise the following YouTube video "
                "transcript in 4-6 concise bullet points that cover the whole video.",
            ),
            ("human", f"Transcript:\n{context}\n\nSummary:"),
        ]
    )
    llm = ChatGroq(model=model)
    chain = prompt | llm | StrOutputParser()
    logger.info(f"Generating video summary | model={model} | sample_chunks={len(sample)}")
    return chain.invoke({})
