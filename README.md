# RAG Assignment – Generative AI with Retrieval-Augmented Generation

This project implements multiple Retrieval-Augmented Generation (RAG) systems using Python and a local LLM via Ollama.

Each system answers questions only using its own document dataset, preventing hallucination and improving accuracy.

## Use Cases
- Technical Documentation Assistant
- Employee Knowledge Assistant
- Support Ticket Assistant
- Legal Document Assistant

## How It Works
1. Documents are split into chunks
2. Chunks converted to vectors (TF-IDF)
3. User query compared using cosine similarity
4. Most relevant chunk sent to LLM
5. LLM answers using only that context


