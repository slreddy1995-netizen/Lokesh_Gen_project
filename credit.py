import pandas as pd
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# ================= CLEAN FUNCTION =================
def clean_data(df):
    df['Tax_type'] = df['Tax_type'].astype(str).str.upper().str.strip()
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    df = df[df['Tax_type'] != 'SHIP']
    df = df[df['amount'].notna() & (df['amount'] != 0)]

    return df


# ================= REMOVE DUPLICATES =================
def remove_duplicates(df):
    return df.drop_duplicates(subset=['invoice_number'])


# ================= DATE STANDARDIZATION =================
def standardize_dates(df):
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce').dt.strftime('%d/%m/%Y')
    df['original_invoice_date'] = pd.to_datetime(df['original_invoice_date'], errors='coerce').dt.strftime('%d/%m/%Y')
    return df


# ================= EXPORT =================
def export_csv(df, path="cleaned_invoice.csv"):
    df.to_csv(path, index=False)
    return path


# ================= PIPELINE =================
def run_pipeline(file_path):
    df = pd.read_csv(file_path)

    df = clean_data(df)
    df = remove_duplicates(df)
    df = standardize_dates(df)

    export_csv(df)

    return df


# ================= EMBEDDINGS =================
def create_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Convert each row to text
    texts = df.astype(str).apply(lambda x: " | ".join(x), axis=1)

    embeddings = model.encode(texts.tolist())

    return embeddings, texts.tolist()


# ================= FAISS INDEX =================
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index


# ================= RETRIEVAL =================
def retrieve_from_vector_db(query, model, index, texts, top_k=3):
    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), top_k)

    results = [texts[i] for i in indices[0]]
    return results


# ================= SIMPLE LLM MOCK =================
def llm_generate(results, query):
    response = f"Query: {query}\n\nRelevant Data:\n"
    for r in results:
        response += f"- {r}\n"
    return response


# ================= STREAMLIT APP =================
st.title("📊 Invoice AI Assistant")

uploaded_file = st.file_uploader("Upload Invoice CSV", type=["csv"])

if uploaded_file:
    df = run_pipeline(uploaded_file)

    st.success("✅ Data cleaned and processed")
    st.dataframe(df.head())

    # Create embeddings + FAISS
    embeddings, texts = create_embeddings(df)
    index = create_faiss_index(embeddings)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    query = st.text_input("💬 Ask about invoices")

    if query:
        results = retrieve_from_vector_db(query, model, index, texts)
        response = llm_generate(results, query)

        st.write(response)