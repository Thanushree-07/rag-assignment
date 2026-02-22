import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text):

    sections = text.split("\n\n")
    chunks = []

    for i in range(len(sections)):
        chunk = sections[i]

        if i + 1 < len(sections):
            chunk += " " + sections[i + 1]

        chunks.append(chunk.strip())

    return chunks


# ---------- RETRIEVE MOST RELEVANT SECTION ----------
def retrieve(query, vectorizer, vectors, chunks):

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, vectors)[0]

    best_score = max(similarities)
    best_index = similarities.argmax()

    print("Similarity:", round(best_score, 3))

    if best_score < 0.20:
        return None

    return chunks[best_index]


# ---------- GENERATE ANSWER USING OLLAMA ----------
def generate_answer(context, question):

    prompt = f"""
You are an employee policy assistant.

TASK:
Answer using ONLY the employee handbook.

RULES:
- Use only given document text.
- Do NOT use outside knowledge.
- If answer not present say:
  Not found in employee handbook.

Employee Handbook:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# ---------- MAIN PROGRAM ----------
document = load_document("employee.txt")

chunks = split_text(document)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(chunks)

print("\nEmployee Policy Assistant Ready\n")

while True:

    question = input("Ask: ")

    if question.lower() == "exit":
        break

    context = retrieve(question, vectorizer, vectors, chunks)

    if context is None:
        print("\nNot found in employee handbook.\n")
        continue

    answer = generate_answer(context, question)

    print("\nAnswer:", answer, "\n")
