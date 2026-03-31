import streamlit as st
import pandas as pd
import numpy as np
import faiss
import ollama
import spacy


# ================= CLEAN AGENT =================
def clean_data(df):
    df['Tax_type'] = df['Tax_type'].astype(str).str.upper().str.strip()
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    df = df[df['Tax_type'] != 'SHIP']
    df = df[df['amount'].notna() & (df['amount'] != 0)]

    return df


# ================= DEDUP AGENT =================
def remove_duplicates(df):
    return df.drop_duplicates(subset=['Invoice_Num'])


# ================= DATE AGENT =================
def standardize_dates(df):
    df['Invoice_date'] = pd.to_datetime(df['Invoice_date'], errors='coerce').dt.strftime('%d/%m/%Y')
    df['Original_Invoice_date'] = pd.to_datetime(df['Original_Invoice_date'], errors='coerce').dt.strftime('%d/%m/%Y')
    return df


# ================= EXPORT AGENT =================
def export_csv(df):
    output_path = "cleaned_invoice.csv"
    df.to_csv(output_path, index=False)
    return output_path


# ================= LOAD EMBEDDING MODEL =================
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_md")


# ================= CREATE EMBEDDINGS =================
def create_embeddings(df, model):
    texts = df.astype(str).apply(lambda x: " | ".join(x), axis=1)

    embeddings = []
    for text in texts:
        doc = model(text)
        embeddings.append(doc.vector)

    return np.array(embeddings), texts.tolist()


# ================= FAISS =================
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index


# ================= RETRIEVAL =================
def retrieve_from_vector_db(query, model, index, texts, top_k=3):
    query_embedding = model(query).vector.reshape(1, -1)
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [texts[i] for i in indices[0]]
    return results


# ================= STREAMLIT UI =================
st.title("📊 Invoice Agentic AI System")

uploaded_file = st.file_uploader("Upload Invoice CSV", type=["csv"])

if uploaded_file:

    # ================= LOAD DATA =================
    df = pd.read_csv(uploaded_file)

    st.subheader("📂 Raw Data")
    st.dataframe(df.head())

    # ================= RUN AGENTS =================
    df = clean_data(df)
    df = remove_duplicates(df)
    df = standardize_dates(df)

    st.subheader("✅ Cleaned Data")
    st.dataframe(df.head())

    # ================= EXPORT =================
    file_path = export_csv(df)
    st.success(f"File saved as: {file_path}")

    # Download button
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=df.to_csv(index=False),
        file_name="cleaned_invoice.csv",
        mime="text/csv"
    )

    # ================= RAG SETUP =================
    model = load_model()
    embeddings, texts = create_embeddings(df, model)
    index = create_faiss_index(embeddings)

    # ================= CHAT MEMORY =================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ================= CHAT INPUT =================
    user_input = st.chat_input("💬 Ask about invoices...")

    if user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant data
        results = retrieve_from_vector_db(user_input, model, index, texts)
        context = "\n".join(results)

        # Build full conversation
        messages = [{"role": "system", "content": "You are an invoice analysis assistant."}]

        for msg in st.session_state.messages:
            messages.append(msg)

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"
        })

        # ================= LLM RESPONSE =================
        response = ollama.chat(
            model="llama3",
            messages=messages
        )

        ai_reply = response['message']['content']

        # Save AI response
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

        with st.chat_message("assistant"):
            st.markdown(ai_reply)