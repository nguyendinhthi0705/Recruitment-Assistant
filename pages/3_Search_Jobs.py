import streamlit as st 
import recruitment_lib as glib 
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
import recruitment_lib as glib 
from langchain.callbacks import StreamlitCallbackHandler

def print_result(st, response):
    try:
        st.dataframe(response['intermediate_steps'][1][1])
        st.subheader("Conclusion:")
        st.write(response['output'])
    except:
        st.write(response['output'])

input_text = st.text_input("Solutions Architect, Technical Lead") 
if input_text: 
    st_callback = StreamlitCallbackHandler(st.container())
    response = glib.search_jobs(input_text, st_callback)
    print_result(st,response)
