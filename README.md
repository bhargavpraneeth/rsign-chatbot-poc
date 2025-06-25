# 📩 RSign Chatbot POC

This project is a Proof of Concept chatbot for RPost's **RSign** product. It answers user queries using official documentation pulled directly from [RSign Help Center](https://help.rpost.com/hc/en-us/categories/18954520975251-RSign).

## 💡 Features

- Scrapes top help articles from RSign
- Uses FAISS for fast document search
- Embeds documents using OpenAI
- GPT-4 generates user-friendly responses
- Streamlit web interface

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/rsign-chatbot-poc.git
cd rsign-chatbot-poc
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
streamlit run app.py
