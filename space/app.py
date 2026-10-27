"""
Chatbot Tim Legal berbasis RAG (HyDE + Ensemble + Parent-Child + Reranker + DuckDuckGo fallback)
Deploy: Hugging Face Space (Gradio SDK, hardware ZeroGPU).
Struktur Space:
    app.py | requirements.txt | README.md
    data/parents.json            <- hasil export dari notebook RAG
    data/faiss_index/            <- hasil FAISS.save_local()
"""
import os
import re
import json
from typing import Any, List

import torch
import spaces
import gradio as gr

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
try:
    from langchain_core.stores import InMemoryStore
except ImportError:
    from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# ------------------------------------------------------------------ Konfigurasi
GRPO_REPO = os.environ.get("LLM_REPO", "putraharifin/llama-3.2-3b-legal-id-GRPO-PGABL-putraharifin25")
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_THRESHOLD = float(os.environ.get("RERANK_THRESHOLD", "0.25"))   # samakan dengan hasil kalibrasi di notebook
ENSEMBLE_TOP_K = 10
HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ------------------------------------------------------------------ Data & retriever
with open(os.path.join(DATA_DIR, "parents.json"), encoding="utf-8") as f:
    parent_docs = [Document(page_content=x["page_content"], metadata=x["metadata"]) for x in json.load(f)]

embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": "cpu"},                      # 1 query pendek, CPU sudah cukup
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local(
    os.path.join(DATA_DIR, "faiss_index"), embeddings, allow_dangerous_deserialization=True
)
docstore = InMemoryStore()
docstore.mset([(p.metadata["parent_id"], p) for p in parent_docs])


class ParentChildRetriever(BaseRetriever):
    """Cari Child Chunk di FAISS, kembalikan Parent Chunk (dedupe, urut relevansi)."""
    vectorstore: Any
    docstore: Any
    k: int = 10
    child_k: int = 40

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        children = self.vectorstore.similarity_search(query, k=self.child_k)
        ids, seen = [], set()
        for c in children:
            pid = c.metadata["parent_id"]
            if pid not in seen:
                seen.add(pid)
                ids.append(pid)
            if len(ids) >= self.k:
                break
        return [p for p in self.docstore.mget(ids) if p is not None]


bm25_retriever = BM25Retriever.from_documents(parent_docs)
bm25_retriever.k = ENSEMBLE_TOP_K
vector_retriever = ParentChildRetriever(vectorstore=vectorstore, docstore=docstore, k=ENSEMBLE_TOP_K)
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])

# ------------------------------------------------------------------ Model (dimuat di level modul untuk ZeroGPU)
tokenizer = AutoTokenizer.from_pretrained(GRPO_REPO, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(GRPO_REPO, torch_dtype=torch.float16, token=HF_TOKEN).to("cuda").eval()

rr_tok = AutoTokenizer.from_pretrained(RERANK_MODEL)
rr_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, torch_dtype=torch.float16).to("cuda").eval()


def llm_generate(messages, max_new_tokens=512, temperature=0.3, top_p=0.9, n=1):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
            top_p=top_p, num_return_sequences=n, use_cache=True, pad_token_id=tokenizer.pad_token_id,
        )
    plen = inputs["input_ids"].shape[1]
    return [tokenizer.decode(o[plen:], skip_special_tokens=True).strip() for o in out]


# ------------------------------------------------------------------ Metadata filter, HyDE, rerank, web
def infer_filter(query):
    m = re.search(r"\b(UU|PP)\s*(?:Nomor|No\.?)?\s*(\d+)(?:\s*Tahun\s*(\d{4}))?", query, flags=re.IGNORECASE)
    if not m:
        return None
    flt = {"jenis": m.group(1).upper(), "nomor": int(m.group(2))}
    if m.group(3):
        flt["tahun"] = int(m.group(3))
    return flt


def apply_filter(docs, flt):
    if not flt:
        return docs
    kept = [d for d in docs if all(d.metadata.get(k) == v for k, v in flt.items())]
    return kept or docs


HYDE_SYSTEM = ("Kamu pakar hukum ketenagakerjaan dan peraturan perundang-undangan Indonesia. "
               "Tulis jawaban hipotesis singkat (2-3 kalimat) seperti kutipan ketentuan peraturan yang relevan "
               "terhadap pertanyaan, dalam Bahasa Indonesia. Jangan gunakan tag <think>.")


def clean_hypothesis(t):
    c = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    c = re.sub(r"</?think>", "", c).strip()
    if len(c) < 20:
        c = re.sub(r"</?think>", "", t).strip()
    return c


def generate_hypotheses(query, n=2):
    outs = llm_generate(
        [{"role": "system", "content": HYDE_SYSTEM}, {"role": "user", "content": query}],
        max_new_tokens=160, temperature=0.8, n=n,
    )
    return [clean_hypothesis(o) for o in outs]


def hyde_ensemble_retrieve(query, flt=None, n_hypo=2):
    hypotheses = generate_hypotheses(query, n=n_hypo)
    candidates, seen = [], set()
    for q in [query] + hypotheses:
        for d in ensemble_retriever.invoke(q):
            if d.page_content not in seen:
                seen.add(d.page_content)
                candidates.append(d)
    return apply_filter(candidates, flt), hypotheses


def rerank(query, docs, top_k=3, batch_size=8):
    scores = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        enc = rr_tok([query] * len(batch), [d.page_content for d in batch], padding=True,
                     truncation="only_second", max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = rr_model(**enc).logits.view(-1).float()
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]


def web_search(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{query} hukum Indonesia", region="id-id", max_results=max_results))
    except Exception as e:
        print("[DuckDuckGo] gagal:", e)
        return []
    return [Document(page_content=r.get("body", ""),
                     metadata={"source": r.get("href", ""), "title": r.get("title", "")}) for r in results]


def retrieve_context(query):
    flt = infer_filter(query)
    candidates, hypotheses = hyde_ensemble_retrieve(query, flt)
    ranked = rerank(query, candidates, top_k=3)
    top1 = ranked[0][1] if ranked else 0.0
    if top1 >= RERANK_THRESHOLD:
        return {"mode": "local", "docs": [d for d, _ in ranked], "scores": [s for _, s in ranked],
                "top1": top1, "hypotheses": hypotheses, "filter": flt}
    return {"mode": "web", "docs": web_search(query), "scores": [s for _, s in ranked],
            "top1": top1, "hypotheses": hypotheses, "filter": flt}


# ------------------------------------------------------------------ Prompt & rendering
RAG_SYSTEM = ("Kamu adalah asisten AI internal Tim Legal yang menjawab pertanyaan hukum secara akurat dan "
              "dapat dipertanggungjawabkan, hanya berdasarkan konteks dokumen yang diberikan. Jangan mengarang pasal.")

RAG_USER_TEMPLATE = """Konteks dokumen:
{context}

Pertanyaan: {question}

Instruksi:
1. Tulis proses berpikir penalaran hukum langkah demi langkah di dalam tag <think>...</think>, sebutkan peraturan/pasal yang relevan dari konteks.
2. Setelah </think>, tulis jawaban final yang logis, ringkas, dan berbahasa Indonesia.
3. Jika konteks tidak memuat jawabannya, katakan bahwa informasinya tidak ditemukan."""


def format_context(res):
    parts = []
    for i, d in enumerate(res["docs"], 1):
        label = d.metadata.get("citation") or d.metadata.get("title") or "web"
        parts.append(f"[{i}] ({label})\n{d.page_content[:1200]}")
    return "\n\n".join(parts) if parts else "(tidak ada konteks ditemukan)"


def split_reasoning(text):
    if "</think>" in text:
        think, final = text.split("</think>", 1)
        return think.replace("<think>", "").strip(), final.strip()
    if "<think>" in text:
        return text.split("<think>", 1)[1].strip(), "_(Jawaban terpotong sebelum selesai. Coba ulangi pertanyaan.)_"
    return "", text.strip()


@spaces.GPU(duration=120)
def run_rag(question):
    question = (question or "").strip()
    if not question:
        return "Silakan tulis pertanyaan terlebih dahulu.", ""
    try:
        res = retrieve_context(question)
        prompt = RAG_USER_TEMPLATE.format(context=format_context(res), question=question)
        raw = llm_generate([{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": prompt}],
                           max_new_tokens=768, temperature=0.3)[0]
    except Exception as e:
        return f"Terjadi kesalahan saat memproses pertanyaan: `{e}`", ""

    think, final = split_reasoning(raw)
    if res["mode"] == "local":
        note = f"Dokumen internal (skor relevansi Top-1: {res['top1']:.2f})"
        sources, seen = [], set()
        for d in res["docs"]:
            c = d.metadata["citation"]
            if c not in seen:
                seen.add(c)
                sources.append(f"- {c}")
    else:
        note = f"Dokumen lokal tidak relevan (skor Top-1: {res['top1']:.2f}), jawaban memakai pencarian internet"
        sources = [f"- [{d.metadata.get('title') or d.metadata['source']}]({d.metadata['source']})"
                   for d in res["docs"]] or ["- (tidak ada hasil pencarian)"]

    quoted = think.replace("\n", "\n> ") if think else "_(model tidak menampilkan penalaran)_"
    answer_md = (f"### 🧠 Proses Penalaran\n> {quoted}\n\n"
                 f"### ✅ Jawaban\n{final}\n\n"
                 f"### 📚 Sumber\n*{note}*\n\n" + "\n".join(sources) +
                 "\n\n---\n*⚠️ Informasi ini bersifat referensi dan bukan nasihat hukum resmi.*")

    debug_md = [f"**Mode:** `{res['mode']}` | **Threshold:** `{RERANK_THRESHOLD}` | **Filter metadata:** `{res['filter']}`", ""]
    for i, h in enumerate(res["hypotheses"], 1):
        debug_md.append(f"**Hipotesis HyDE {i}:** {h}\n")
    debug_md.append("**Skor reranker (Top-3):** " + ", ".join(f"{s:.3f}" for s in res["scores"]))
    return answer_md, "\n".join(debug_md)


# ------------------------------------------------------------------ UI
EXAMPLES = [
    "Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya berhak dapat uang lembur?",
    "Berapa lama maksimal perjanjian kerja waktu tertentu (PKWT) menurut PP 35 Tahun 2021?",
    "Apa saja hak pekerja yang di-PHK menurut peraturan yang berlaku?",
    "Bagaimana resep rendang yang enak?",
]

with gr.Blocks(title="Chatbot Tim Legal RAG") as demo:
    gr.Markdown(
        "# ⚖️ Chatbot Tim Legal berbasis RAG\n"
        "Tanya seputar **UU No. 6/2023, PP No. 5/2021, PP No. 35/2021, dan PP No. 51/2023**. "
        "Model: Llama-3.2-3B hasil SFT + GRPO (penalaran `<think>`), retrieval: HyDE + BM25/FAISS + Parent-Child + Cross-Encoder."
    )
    q = gr.Textbox(label="Pertanyaan", lines=2, placeholder="Contoh: Apakah saya berhak mendapat uang lembur?")
    btn = gr.Button("Tanya", variant="primary")
    out = gr.Markdown()
    with gr.Accordion("Detail retrieval (HyDE & skor reranker)", open=False):
        dbg = gr.Markdown()
    gr.Examples(examples=EXAMPLES, inputs=q)
    btn.click(run_rag, inputs=q, outputs=[out, dbg])
    q.submit(run_rag, inputs=q, outputs=[out, dbg])

demo.queue().launch()
