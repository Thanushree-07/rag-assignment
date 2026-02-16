import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- LOAD DOCUMENT ----------
def load_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------- SEMANTIC CHUNKING + OVERLAP ----------
def split_text(text):

    sections = text.split("\n\n")  # split by paragraph meaning

    chunks = []

    for i in range(len(sections)):
        chunk = sections[i]

        # overlap with next section
        if i + 1 < len(sections):
            chunk += " " + sections[i + 1]

        chunks.append(chunk.strip())

    return chunks


# ---------- RETRIEVE ----------
def retrieve(query, vectorizer, vectors, chunks):

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, vectors)[0]

    best_score = max(similarities)
    best_index = similarities.argmax()

    print("Similarity:", best_score)

    # relevance filter
    if best_score < 0.25:
        return None

    return chunks[best_index]


# ---------- GENERATE (EXTRACTIVE MODE) ----------
def generate_answer(context, question):

    prompt = f"""
You are a handbook assistant.

TASK:
Extract the exact answer from the handbook.

RULES:
- Only copy the sentence from the handbook.
- Do NOT explain.
- Do NOT calculate.
- Do NOT add information.
- If answer not present, say:
  Answer not found in handbook.

Handbook:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="phi",
        messages=[{"role":"user","content":prompt}]
    )

    return response["message"]["content"]


# ---------- MAIN ----------
document = load_document("employee.txt")

chunks = split_text(document)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(chunks)

print("\nEmployee Assistant Ready\n")

while True:
    question = input("Ask: ")

    if question.lower() == "exit":
        break

    context = retrieve(question, vectorizer, vectors, chunks)

    if context is None:
        print("\nAnswer: Answer not found in handbook.\n")
        continue

    answer = generate_answer(context, question)

    print("\nAnswer:", answer, "\n")
