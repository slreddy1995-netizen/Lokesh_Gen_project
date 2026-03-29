import streamlit as st
import pandas as pd
import ollama

# ================= CLEAN FUNCTION =================
def clean_invoice_data(df):

    # Normalize Tax_type
    df['Tax_type'] = df['Tax_type'].astype(str).str.strip().str.upper()

    # Convert amount
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Remove SHIP
    df = df[df['Tax_type'] != 'SHIP']

    # Remove zero/NaN
    df = df[df['amount'].notna() & (df['amount'] != 0)]

    # Remove duplicates
    df = df.drop_duplicates(subset=['Invoice_Num'])

    # Convert dates (input = dd-mm-yyyy)
    df['Invoice_date'] = pd.to_datetime(df['Invoice_date'], format='%d-%m-%Y', errors='coerce')
    df['Original_Invoice_date'] = pd.to_datetime(df['Original_Invoice_date'], format='%d-%m-%Y', errors='coerce')

    # Format output (dd/mm/yyyy)
    df['Invoice_date'] = df['Invoice_date'].dt.strftime('%d/%m/%Y')
    df['Original_Invoice_date'] = df['Original_Invoice_date'].dt.strftime('%d/%m/%Y')

    return df


# ================= AI FUNCTION =================
def explain_row(row):
    prompt = f"""
    Invoice {row['Invoice_Num']}:
    Amount: {row['amount']}
    Tax Type: {row['Tax_type']}
    Explain if this invoice looks correct or not.
    """

    response = ollama.chat(
        model='llama3',
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']


# ================= STREAMLIT UI =================
st.title("📊 Invoice Reconciliation AI")

uploaded_file = st.file_uploader("Upload CSV or Excel")

if uploaded_file:

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Original Data")
    st.dataframe(df)

    if st.button("Run Cleaning"):

        cleaned_df = clean_invoice_data(df)

        st.subheader("Cleaned Data")
        st.dataframe(cleaned_df)

        # Download option
        st.download_button(
            label="Download Cleaned Report",
            data=cleaned_df.to_csv(index=False),
            file_name="cleaned_report.csv",
            mime="text/csv"
        )

        # Optional AI explanation
        if len(cleaned_df) > 0:
            if st.button("Explain First Row"):
                explanation = explain_row(cleaned_df.iloc[0])
                st.write(explanation)