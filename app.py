import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD DATASET
books = pd.read_csv(
    "new_dataset.csv",
    encoding="latin1"
)

books = books.head(1000)

# LOAD EMBEDDINGS
with open("embeddings.pkl", "rb") as f:
    book_embeddings = pickle.load(f)

# MODEL WILL LOAD ONLY WHEN NEEDED
model = None


@app.get("/")
def home():
    return {
        "message": "NovelCompass AI Running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/recommend")
def recommend(user_query: str):

    global model

    # Import only when endpoint is called
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    query_embedding = model.encode([user_query])

    similarities = cosine_similarity(
        query_embedding,
        book_embeddings
    )[0]

    top_indices = similarities.argsort()[-5:][::-1]

    results = []

    for i in top_indices:

        results.append({

            "title": str(
                books.iloc[i]["title"]
            ),

            "author": str(
                books.iloc[i]["author"]
            ),

            "genre": str(
                books.iloc[i]["genres"]
            ),

            "pages": str(
                books.iloc[i]["pages"]
            ),

            "rating": str(
                books.iloc[i]["rating"]
            ),

            "description": str(
                books.iloc[i]["description"]
            ),

            "cover": str(
                books.iloc[i]["coverImg"]
            ),

            "score": round(
                float(similarities[i] * 100),
                2
            )
        })

    return results