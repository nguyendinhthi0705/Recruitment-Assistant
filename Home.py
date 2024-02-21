import streamlit as st 
import recruitment_lib as glib 
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Home")


if 'chat_history' not in st.session_state: 
    st.session_state.chat_history = [] 


for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"]) 

st.markdown("Ask me anything as below samples:") 
st.markdown("Viết CV dành cho software developers có 5 năm kinh nghiệm trong lập trình web với reactjs và .NET core") 
st.markdown("Liệt kê 10 câu hỏi dành cho lập trình viên React") 
st.markdown("Top 10 questions for JavaScript") 

input_text = st.text_input("") 
if input_text: 
    st_callback = StreamlitCallbackHandler(st.container())
    with st.chat_message("user"): 
        st.markdown(input_text) 
    st.session_state.chat_history.append({"role":"user", "text":input_text}) 
    
    chat_response = glib.get_rag_chat_response(input_text, st_callback) 
    
    with st.chat_message("assistant"): 
        st.markdown(chat_response) 
    
    st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) 