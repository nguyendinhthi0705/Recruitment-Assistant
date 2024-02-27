import streamlit as st 
import recruitment_lib as glib 
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Rewrite a Resume")


input_text = st.text_area("Input your whole or apart of your resume") 
if input_text: 
    st_callback = StreamlitCallbackHandler(st.container())
    chat_response = glib.rewrite_resume(input_text, st_callback) 
    with st.chat_message("assistant"): 
        st.markdown(chat_response) 
    
