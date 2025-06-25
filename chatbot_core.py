import os
import openai
import faiss
import numpy as np
from requests_html import HTMLSession

openai.api_key = os.getenv("OPENAI_API_KEY")  # Securely read from Streamlit secrets

# Step 1: Fetch article content from RSign help center
def fetch_articles(limit=3):
    session = HTMLSession()
    base_url = "https://help.rpost.com"
    category_url = f"{base_url}/hc/en-us/categories/18954520975251-RSign"

    r = session.get(category_url)
    articles = r.html.find("a.article-list-link")
    data = []

    for a in articles[:limit]:
        url = base_url + a.attrs["href"]
        title = a.text.strip()
        try:
            res = session.get(url)
            content_div = res.html.find("div.article-body", first=True)
            content = content_div.text.strip() if content_div else ""
            data.append({"title": title, "url": url, "content": content})
        except Exception as e:
            print(f"Error reading {url}: {e}")
            continue

    return data

# Step 2: Get embedding for a text chunk
def get_embedding(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype='float32')

# Step 3: Create FAISS vector store
def create_vector_store(docs):
    dimension = 1536
    index = faiss.IndexFlatL2(dimension)
    chunks = []
    metadata = []

    for doc in docs:
        content = doc["content"]
        chunked = [content[i:i+800] for i in range(0, len(content), 800)]
        for chunk in chunked:
            try:
                emb = get_embedding(chunk)
                index.add(np.array([emb]))
                chunks.append(chunk)
                metadata.append(doc)
            except Exception as e:
                print(f"Embedding failed: {e}")
                continue

    return index, chunks, metadata

# Step 4: Find similar document chunks for user query
def search_docs(query, index, chunks, k=3):
    emb = get_embedding(query)
    D, I = index.search(np.array([emb]), k)
    return [chunks[i] for i in I[0]]

# Step 5: Generate GPT-4 answer from context
def generate_answer(user_q, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an expert assistant for RPost's RSign product.
Based on the following official documentation, answer the user's question accurately.

Documentation:
{context}

User Question: {user_q}
"""
    res = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return res.choices[0].message.content.strip()
