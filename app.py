import streamlit as st
from elasticsearch import Elasticsearch
import urllib3
import PyPDF2
import io
import re
import os
from dotenv import load_dotenv

load_dotenv()

urllib3.disable_warnings()

# Connect to ES using .env variables
es = Elasticsearch(
    os.getenv("ES_HOST"),
    basic_auth=(os.getenv("ES_USER"), os.getenv("ES_PASSWORD")),
    verify_certs=False,
    request_timeout=120
)

st.title("📄 PDF Semantic Search with ELSER")
st.markdown("Upload a PDF → Vectorize with ELSER → Search semantically!")

# ── SIDEBAR ──────────────────────────────────────
st.sidebar.header("⚙️ Status")
try:
    stats = es.ml.get_trained_models_stats(model_id=".elser_model_2")
    state = stats['trained_model_stats'][0]['deployment_stats']['state']
    st.sidebar.success(f"🤖 ELSER: {state}")
except:
    st.sidebar.error("❌ ELSER not running!")

# ── CHUNKING FUNCTION ─────────────────────────────
def split_into_chunks(text, max_chars=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ── UPLOAD PDF ────────────────────────────────────
st.header("1️⃣ Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    st.success(f"✅ Extracted {len(text)} characters from PDF!")
    st.text_area("📄 Extracted Text Preview", text[:500] + "...", height=150)

    chunks = split_into_chunks(text)
    st.info(f"📦 Will be split into {len(chunks)} clean sentence-based chunks")

    with st.expander("👀 Preview Chunks"):
        for i, chunk in enumerate(chunks[:3]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(chunk)
            st.divider()

    if st.button("🚀 Vectorize & Store in Elasticsearch"):
        if es.indices.exists(index="pdf-index"):
            es.indices.delete(index="pdf-index")
        es.indices.create(index="pdf-index", body={
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "content_embedding": {"type": "sparse_vector"}
                }
            }
        })

        es.ingest.put_pipeline(id="elser-pipeline", body={
            "processors": [{
                "inference": {
                    "model_id": ".elser_model_2",
                    "input_output": [{
                        "input_field": "content",
                        "output_field": "content_embedding"
                    }]
                }
            }]
        })

        progress = st.progress(0)
        status = st.empty()

        for i, chunk in enumerate(chunks):
            es.index(
                index="pdf-index",
                pipeline="elser-pipeline",
                document={"content": chunk}
            )
            progress.progress((i + 1) / len(chunks))
            status.text(f"⏳ Uploading chunk {i+1}/{len(chunks)}...")

        st.success(f"🎉 PDF vectorized! {len(chunks)} chunks stored in Elasticsearch!")

        st.header("🔢 Sample Vectors Stored")
        results = es.search(index="pdf-index", body={
            "query": {"match_all": {}},
            "size": 2
        })
        for hit in results['hits']['hits']:
            with st.expander(f"📄 Chunk: {hit['_source']['content'][:100]}..."):
                st.write("**Text:**", hit['_source']['content'])
                st.write("**Vector (top 5 tokens):**")
                vector = hit['_source'].get('content_embedding', {})
                top5 = dict(sorted(vector.items(),
                           key=lambda x: x[1], reverse=True)[:5])
                st.json(top5)

# ── SEARCH ───────────────────────────────────────
st.header("2️⃣ Search Semantically")
query = st.text_input("🔍 Ask anything about the PDF...")

if query and st.button("Search"):
    try:
        results = es.search(index="pdf-index", body={
            "query": {
                "sparse_vector": {
                    "field": "content_embedding",
                    "inference_id": ".elser_model_2",
                    "query": query
                }
            }
        })

        st.subheader("📊 Results:")
        max_score = results['hits']['max_score']

        for i, hit in enumerate(results['hits']['hits']):
            match_percent = (hit['_score'] / max_score) * 100

            if match_percent >= 80:
                emoji = "🟢"
            elif match_percent >= 50:
                emoji = "🟡"
            else:
                emoji = "🔴"

            with st.expander(f"{emoji} Result {i+1} — Match: {match_percent:.1f}%"):
                st.progress(match_percent / 100)
                st.write(hit['_source']['content'])

    except Exception as e:
        st.error(f"Error: {e}")