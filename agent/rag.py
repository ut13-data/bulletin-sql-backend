"""
Answers from the two business documents (RAG), used for definition and
business-context questions that the metric catalog doesn't cover.
"""
from functools import lru_cache

import requests
from langchain_core.embeddings import Embeddings

from agent import llm
from agent.config import DOCS_DIR, SIMILARITY_DISTANCE_THRESHOLD, get_secret


class HFAPIEmbeddings(Embeddings):
    """Embeddings via the hosted HF Inference API (no local PyTorch needed)."""

    def __init__(self, api_token: str):
        self.api_url = ("https://router.huggingface.co/hf-inference/models/"
                        "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction")
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def _embed(self, texts):
        response = requests.post(self.api_url, headers=self.headers,
                                 json={"inputs": texts, "options": {"wait_for_model": True}}, timeout=60)
        response.raise_for_status()
        return response.json()

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    architecture = (DOCS_DIR / "Balaji_Pharma_Database_Architecture.md").read_text(encoding="utf-8")
    business = (DOCS_DIR / "Balaji_Pharma_Business_Definition.md").read_text(encoding="utf-8")

    # Architecture doc: real boundaries at ## and ###. Business doc: ## only.
    arch_chunks = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section"), ("###", "subsection")]).split_text(architecture)
    biz_chunks = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")]).split_text(business)

    content, metadatas = [], []
    for chunk in arch_chunks:
        content.append(chunk.page_content)
        metadatas.append({**chunk.metadata, "source": "Database Architecture"})
    for chunk in biz_chunks:
        content.append(chunk.page_content)
        metadatas.append({**chunk.metadata, "source": "Business Definition"})

    return FAISS.from_texts(content, HFAPIEmbeddings(api_token=get_secret("HUGGINGFACEHUB_API_TOKEN")),
                            metadatas=metadatas)


def answer_from_docs(question: str) -> dict:
    try:
        results = get_vectorstore().similarity_search_with_score(question, k=5)
    except Exception as e:
        print(f"DEBUG: RAG unavailable: {e}")
        return {"found": False, "explanation": "The business documents are unavailable right now, please try again.",
                "sources": []}

    relevant = [(doc, score) for doc, score in results if score <= SIMILARITY_DISTANCE_THRESHOLD][:3]
    if not relevant:
        return {"found": False, "sources": [],
                "explanation": "I couldn't find anything about that in Balaji Pharma's business documentation."}

    context = "\n\n".join(doc.page_content for doc, _ in relevant)
    answer = llm.text(
        f"Context:\n\n{context}\n\nUsing ONLY the context above, answer the question in 2-5 sentences. "
        f"If the answer isn't in the context, reply exactly: I don't know.\n\nQuestion: {question}",
        max_tokens=600,
    )
    if not answer or answer.strip().lower().replace("’", "'").startswith("i don't know"):
        return {"found": False, "sources": [],
                "explanation": "I couldn't find anything about that in Balaji Pharma's business documentation."}

    sources = [{"document": d.metadata.get("source", "Unknown"), "section": d.metadata.get("section", ""),
                "subsection": d.metadata.get("subsection", "")} for d, _ in relevant]
    return {"found": True, "explanation": answer, "sources": sources}
