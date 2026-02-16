import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_document(path):
    with open(path,"r",encoding="utf-8") as f:
        return f.read()


def split_text(text):
    return text.split("\n\n")


def retrieve(query, vectorizer, vectors, chunks):

    q = vectorizer.transform([query])
    sims = cosine_similarity(q, vectors)[0]

    best = sims.argmax()

    if max(sims) < 0.2:
        return None

    return chunks[best]


def generate(context, question):

    prompt=f"""
Answer ONLY using ticket resolution.
If not found say:
Answer not found in tickets.

Ticket:
{context}

Question:
{question}
"""

    res = ollama.chat(
        model="phi",
        messages=[{"role":"user","content":prompt}]
    )

    return res["message"]["content"]


doc = load_document("tickets.txt")
chunks = split_text(doc)

vec = TfidfVectorizer()
vectors = vec.fit_transform(chunks)

print("Support Assistant Ready\n")

while True:
    q = input("Ask: ")

    if q=="exit":
        break

    ctx = retrieve(q, vec, vectors, chunks)

    if ctx is None:
        print("Answer not found in tickets\n")
        continue

    print(generate(ctx,q),"\n")
