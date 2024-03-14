import streamlit as st 
import recruitment_lib as glib 
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
import recruitment_lib as glib 
from langchain.callbacks import StreamlitCallbackHandler


st.set_page_config(page_title="Questions on Resume")

uploaded_file = st.file_uploader("Upload your resume PDF")
docs = []

st.markdown("Ask me anything as below samples:") 
st.markdown("Does this resume has experience on Reactjs or Angular?") 
st.markdown("Does this resume has strong experience on backend?") 
st.markdown("Hồ sơ này có nhiều kĩ năng .NET không?") 
st.markdown("Does this resume has strong experience on AWS?") 

input_text = st.text_input("Your question!") 
if uploaded_file is not None and input_text:
    st_callback = StreamlitCallbackHandler(st.container())
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        docs.append(page.extract_text())
    
    response = glib.query_resume(input_text, docs, st_callback)

   