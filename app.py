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

# ── ELASTICSEARCH CONNECTION ─────────────────────
es = Elasticsearch(
    os.getenv("ES_HOST"),
    basic_auth=(os.getenv("ES_USER"), os.getenv("ES_PASSWORD")),
    verify_certs=False,
    request_timeout=120
)

st.title("📄 PDF Semantic Search + JSON Extraction (ELSER)")
st.markdown("Upload PDF → Vectorize → Search → Extract Structured JSON")

# ── SIDEBAR ─────────────────────────────────────
st.sidebar.header("⚙️ Status")
try:
    stats = es.ml.get_trained_models_stats(model_id=".elser_model_2")
    state = stats['trained_model_stats'][0]['deployment_stats']['state']
    st.sidebar.success(f"🤖 ELSER: {state}")
except:
    st.sidebar.error("❌ ELSER not running!")

# ── CHUNKING FUNCTION ────────────────────────────
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

# Removed default_schema as the LLM will generate its own schema dynamically

# ── SEMANTIC EXTRACTION USING ELSER ──────────────
def semantic_extract(desc):
    results = es.search(index="pdf-index", body={
        "query": {
            "sparse_vector": {
                "field": "content_embedding",
                "inference_id": ".elser_model_2",
                "query": desc
            }
        },
        "size": 3
    })

    texts = [hit['_source']['content'] for hit in results['hits']['hits']]
    return " ".join(texts)

# ── JSON GENERATOR (GROQ) ────────────────────────
import json
from groq import Groq

def generate_json_with_llm(full_text):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is missing from .env! Please add it.")
    
    client = Groq(api_key=groq_key)
    
    # Truncate to approx 8,000 tokens to safely stay under Groq's 12K Tier 1 RPM limits
    safe_text = full_text[:35000]
    
    prompt = f"""
You are an expert data extractor. You must read the provided PDF text and automatically extract all the core information, policies, values, names, and important details into a highly-structured, comprehensive JSON object.

Invent the most logical and descriptive field names (keys) based on the document's contents. For example, if it's a college document, include 'college_name', 'core_values', 'policies'. If it is a hospital document, include 'hospital_name', 'visiting_hours', etc.

PDF Text content:
{safe_text}

Do NOT include markdown formatting (like ```json). Return ONLY the raw valid JSON object mapping the dynamically generated field names to their values.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    # Clean up response text if necessary
    res_text = response.choices[0].message.content.strip()
    if res_text.startswith("```json"):
        res_text = res_text[7:]
    if res_text.endswith("```"):
        res_text = res_text[:-3]
        
    return json.loads(res_text.strip())

# ── UPLOAD PDF ───────────────────────────────────
st.header("1️⃣ Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    st.session_state["pdf_text"] = text

    st.success(f"✅ Extracted {len(text)} characters!")
    st.text_area("Preview", text[:500] + "...", height=150)

    chunks = split_into_chunks(text)
    st.info(f"📦 {len(chunks)} chunks created")

    if st.button("🚀 Vectorize & Store"):
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

        for i, chunk in enumerate(chunks):
            es.index(
                index="pdf-index",
                pipeline="elser-pipeline",
                document={"content": chunk}
            )
            progress.progress((i + 1) / len(chunks))

        st.success("🎉 PDF processed and stored!")

# ── SEARCH ───────────────────────────────────────
st.header("2️⃣ Semantic Search")
query = st.text_input("Ask something...")

if query and st.button("Search"):
    results = es.search(index="pdf-index", body={
        "query": {
            "sparse_vector": {
                "field": "content_embedding",
                "inference_id": ".elser_model_2",
                "query": query
            }
        }
    })

    for hit in results['hits']['hits']:
        st.write(hit['_source']['content'])

# ── JSON EXTRACTION ──────────────────────────────
st.header("3️⃣ Generate Structured JSON (Auto-Schema)")

st.markdown("🎯 **The AI will automatically invent the schema and extract data directly from the entire PDF context.**")

if st.button("🧠 Extract JSON with Groq"):
    if "pdf_text" not in st.session_state:
        st.warning("Please upload a PDF first!")
    else:
        try:
            with st.spinner("Analyzing text and dynamically creating structure with Groq..."):
                output = generate_json_with_llm(st.session_state["pdf_text"])
                st.success("✅ Extraction complete!")
                st.json(output)
        except Exception as e:
            st.error(f"Error: {e}")