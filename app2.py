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
def split_into_chunks(text, max_chars=300):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    
    for s in sentences:
        s_clean = s.replace('\n', ' ').strip()
        if len(s_clean) > 5:
            chunks.append(s_clean)
            
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

# ── JSON EXTRACTOR (ELSER ONLY) ──────────────────
import json

def extract_from_schema_def(schema_array):
    """Processes an array of field definition objects using pure ELSER."""
    result = {}
    if not isinstance(schema_array, list):
        return result
        
    for item in schema_array:
        field = item.get("field")
        field_type = item.get("type", "string")
        desc = item.get("desc", field)
        
        if not field:
            continue
            
        if field_type == "object":
            props = item.get("properties", [])
            result[field] = extract_from_schema_def(props)
            
        elif field_type == "array":
            hits = es.search(index="pdf-index", body={
                "query": {
                    "sparse_vector": {
                        "field": "content_embedding",
                        "inference_id": ".elser_model_2",
                        "query": desc
                    }
                },
                "size": 3
            })
            if hits['hits']['hits']:
                result[field] = [h['_source']['content'] for h in hits['hits']['hits']]
            else:
                result[field] = []
                
        elif field_type in ["boolean", "bool"]:
            hits = es.search(index="pdf-index", body={
                "query": {
                    "sparse_vector": {
                        "field": "content_embedding",
                        "inference_id": ".elser_model_2",
                        "query": desc
                    }
                },
                "size": 1
            })
            if hits['hits']['hits']:
                result[field] = hits['hits']['hits'][0]['_score'] > 5.0
            else:
                result[field] = False
                
        else: # string
            hits = es.search(index="pdf-index", body={
                "query": {
                    "sparse_vector": {
                        "field": "content_embedding",
                        "inference_id": ".elser_model_2",
                        "query": desc
                    }
                },
                "size": 1
            })
            if hits['hits']['hits']:
                result[field] = hits['hits']['hits'][0]['_source']['content']
            else:
                result[field] = ""
                
    return result


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
st.header("3️⃣ Extract Structured JSON (ELSER Only)")

st.markdown("🎯 **Define field definitions. The system uses ELSER semantic search to find content based on descriptions.**")

default_schema = """[
  {
    "field": "college",
    "type": "string",
    "desc": "Formal name of the college, university, or institution ( St. Paul's College)"
  },
  {
    "field": "content",
    "type": "array",
    "desc": "Code of Conduct and Divyangjan Policy"
  },
  {
    "field": "core_values",
    "type": "array",
    "desc": "Core values such as Excellence and Global Competency"
  },
  {
    "field": "professional_commitment",
    "type": "object",
    "desc": "Professional commitment details",
    "properties": [
      {
        "field": "dedication",
        "type": "boolean",
        "desc": "Dedication to profession"
      },
      {
        "field": "on_time",
        "type": "boolean",
        "desc": "Punctuality and on time"
      },
      {
        "field": "value_based",
        "type": "boolean",
        "desc": "Value based education"
      }
    ]
  }
]"""

user_schema_str = st.text_area("Extraction Schema (Field Definitions)", value=default_schema, height=450)

if st.button("🧠 Extract JSON directly with ELSER"):
    if "pdf_text" not in st.session_state:
        st.warning("Please upload and vectorize a PDF first!")
    else:
        try:
            schema_array = json.loads(user_schema_str)
            with st.spinner("Extracting data semantically using ELSER (No LLM)..."):
                output = extract_from_schema_def(schema_array)
                st.success("✅ Extraction complete!")
                st.json(output)
        except json.JSONDecodeError:
            st.error("Invalid JSON format in the schema. Must be a JSON array.")
        except Exception as e:
            st.error(f"Error: {e}")