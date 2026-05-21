from sentence_transformers import SentenceTransformer

import pandas as pd
import pickle


books = pd.read_csv(
    "new_dataset.csv",
    encoding="latin1"
)

books = books.head(1000)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = (

    books["title"].astype(str)

    + " "

    + books["description"].astype(str)

    + " "

    + books["genres"].astype(str)
)

embeddings = model.encode(
    texts.tolist(),
    show_progress_bar=True
)

with open(
    "embeddings.pkl",
    "wb"
) as f:

    pickle.dump(
        embeddings,
        f
    )

print(
    "Embeddings created successfully!"
)