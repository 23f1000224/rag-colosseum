
import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the 50 passages
with open("passages.json", "r") as f:
    passages = json.load(f)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = passages
embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.post("/query")
def query(data: Query):
    question = data.question.strip()

    if not question:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    q_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    scores = np.dot(embeddings, q_embedding)

    k = max(1, min(data.top_k, 10))
    indices = np.argsort(scores)[::-1][:k]

    best_score = float(scores[indices[0]])

    # Abstain if retrieval confidence is too low
    if best_score < 0.38:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.1
        }

    sources = [int(i) for i in indices]
    selected_scores = [
        round(float(scores[i]), 4) for i in indices
    ]

    # Return relevant source text as the factual answer
    answer = " ".join(texts[i] for i in sources)

    return {
        "answer": answer,
        "sources": sources,
        "scores": selected_scores,
        "confidence": round(best_score, 4)
    }

@app.get("/")
def home():
    return {"status": "RAG API is running"}
