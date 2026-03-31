import streamlit as st
import pandas as pd

# ================= CLEAN FUNCTION =================
def clean_invoice_data(df):

    df['Tax_type'] = df['Tax_type'].astype(str).str.strip().str.upper()
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    df = df[df['Tax_type'] != 'TAX']
    df = df[df['amount'].notna() & (df['amount'] != 0)]
    df = df.drop_duplicates(subset=['Invoice_Num'])

    df['Invoice_date'] = pd.to_datetime(df['Invoice_date'], format='%d-%m-%Y', errors='coerce')
    df['Original_Invoice_date'] = pd.to_datetime(df['Original_Invoice_date'], format='%d-%m-%Y', errors='coerce')

    df['Invoice_date'] = df['Invoice_date'].dt.strftime('%d/%m/%Y')
    df['Original_Invoice_date'] = df['Original_Invoice_date'].dt.strftime('%d/%m/%Y')

    return df


# ================= STREAMLIT UI =================
st.title("📊 Invoice Reconciliation Tool")

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

        # KPIs (Dashboard)
        st.metric("Total Records", len(df))
        st.metric("After Cleaning", len(cleaned_df))
        st.metric("Removed Records", len(df) - len(cleaned_df))

        # Download
        st.download_button(
            label="Download Report",
            data=cleaned_df.to_csv(index=False),
            file_name="cleaned_report.csv",
            mime="text/csv"
        )