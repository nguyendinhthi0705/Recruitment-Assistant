import streamlit as st 
import recruitment_lib as glib 
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
import recruitment_lib as glib 
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Scan Resume")

uploaded_file = st.file_uploader("Upload your resume PDF")
docs = []

if uploaded_file is not None:
    st_callback = StreamlitCallbackHandler(st.container())
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        docs.append(page.extract_text())

    st.write(docs)
   