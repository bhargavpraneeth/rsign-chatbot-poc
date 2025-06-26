import os
import faiss
import openai
import numpy as np
from requests_html import HTMLSession

openai.api_key = "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"
# Step 1: Fetch articles from RSign Help Center
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
            print(f"[Warning] Skipped article due to error: {e}")
            continue

    return data

# Step 2: Create embedding using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return np.array(response.data[0].embedding, dtype='float32')

# Step 3: Build FAISS index for fast similarity search
def create_vector_store(docs):
    dimension = 1536
    index = faiss.IndexFlatL2(dimension)
    chunks = []
    metadata = []

    for doc in docs:
        content = doc["content"]
        # Split into 800-character chunks
        chunked = [content[i:i+800] for i in range(0, len(content), 800)]
        for chunk in chunked:
            try:
                emb = get_embedding(chunk)
                index.add(np.array([emb]))
                chunks.append(chunk)
                metadata.append(doc)
            except Exception as e:
                print(f"[Warning] Embedding failed: {e}")
                continue

    return index, chunks, metadata

# Step 4: Find top-k similar chunks
def search_docs(query, index, chunks, k=3):
    emb = get_embedding(query)
    D, I = index.search(np.array([emb]), k)
    return [chunks[i] for i in I[0]]

# Step 5: Generate GPT-4 answer based on top chunks
def generate_answer(user_q, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant trained on official RPost RSign documentation.

Answer the user's question using the context below:

Context:
{context}

User Question: {user_q}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()
