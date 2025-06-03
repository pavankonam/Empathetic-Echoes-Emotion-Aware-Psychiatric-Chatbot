import sys
import os

# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import streamlit as st
from src.inference import EmpatheticChatbot

st.set_page_config(page_title="🧠 Empathetic Echoes", layout="centered")

@st.cache_resource
def load_chatbot():
    return EmpatheticChatbot()

chatbot = load_chatbot()

EMOTIONS = [
    "sentimental", "worried", "excited", "nervous", "lonely",
    "sad", "terrified", "hopeful", "ashamed", "grateful"
]

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Select Emotional Context")
    selected_emotion = st.selectbox("Choose or type your emotion:", options=EMOTIONS)
    st.markdown("---")
    st.markdown("🧠 *Empathetic Echoes - Psychiatric Chatbot*")

st.title("🧠 Empathetic Echoes")
st.markdown("### A psychiatric chatbot that responds with empathy and emotional awareness.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking empathetically..."):
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]])
            response = chatbot.generate_response(selected_emotion, prompt, chat_history)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []