import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_document(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def split_text(text, chunk_size=40):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0,len(words),chunk_size)]


def retrieve(query, vectorizer, vectors, chunks):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, vectors)[0]

    best_score = max(similarities)
    best_index = similarities.argmax()

    print("Similarity:", best_score)

    if best_score < 0.30:
        return None

    return chunks[best_index]


def generate_answer(context, question):

    prompt = f"""
Answer ONLY using the context below.
If answer is not in context, say:
Answer not found in documentation.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="phi",
        messages=[{"role":"user","content":prompt}]
    )

    return response["message"]["content"]


document = load_document("docs.txt")

chunks = split_text(document)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(chunks)

print("\nRAG Assistant Ready\n")

while True:
    question = input("Ask: ")

    if question.lower()=="exit":
        break

    context = retrieve(question, vectorizer, vectors, chunks)

    if context is None:
        print("\nAnswer: Answer not found in documentation.\n")
        continue

    answer = generate_answer(context, question)

    print("\nAnswer:", answer, "\n")
