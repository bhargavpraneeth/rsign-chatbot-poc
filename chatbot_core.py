import openai
import faiss
import numpy as np
from requests_html import HTMLSession
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Scrape top 3 articles from RSign
def fetch_articles(limit=3):
    session = HTMLSession()
    base_url = "https://help.rpost.com"
    cat_url = f"{base_url}/hc/en-us/categories/18954520975251-RSign"
    r = session.get(cat_url)
    r.html.render(timeout=20)

    articles = r.html.find("a.article-list-link")
    data = []

    for a in articles[:limit]:
        url = base_url + a.attrs["href"]
        title = a.text.strip()
        res = session.get(url)
        res.html.render(timeout=20)
        content_div = res.html.find("div.article-body", first=True)
        content = content_div.text.strip() if content_div else ""
        data.append({"title": title, "url": url, "content": content})
    return data

# Step 2: Embed text
def get_embedding(text):
    res = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(res.data[0].embedding, dtype="float32")

# Step 3: Create FAISS vector index
def create_vector_store(docs):
    dimension = 1536
    index = faiss.IndexFlatL2(dimension)
    chunks, metadata = [], []

    for doc in docs:
        content = doc["content"]
        chunked = [content[i:i+800] for i in range(0, len(content), 800)]
        for chunk in chunked:
            emb = get_embedding(chunk)
            index.add(np.array([emb]))
            chunks.append(chunk)
            metadata.append(doc)
    return index, chunks, metadata

# Step 4: Search docs
def search_docs(query, index, chunks, k=3):
    emb = get_embedding(query)
    D, I = index.search(np.array([emb]), k)
    return [chunks[i] for i in I[0]]

# Step 5: GPT answer
def generate_answer(user_q, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a support assistant for RPost's RSign product. Based on the official documentation below, answer the user's question truthfully.

Documentation:
{context}

User Question: {user_q}
"""
    res = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content.strip()
