import streamlit as st
from chatbot_core import fetch_articles, create_vector_store, search_docs, generate_answer

st.set_page_config(page_title="RSign Chatbot POC", layout="centered")
st.title("📩 RSign Chatbot (POC)")
st.markdown("Ask any question related to RSign and get an instant answer from official documentation.")

with st.spinner("📄 Loading documentation..."):
    articles = fetch_articles()
    index, chunks, _ = create_vector_store(articles)

user_q = st.text_input("🧠 Ask your question about RSign:")

if user_q:
    with st.spinner("🤖 Thinking..."):
        matches = search_docs(user_q, index, chunks)
        response = generate_answer(user_q, matches)
        st.success(response)
