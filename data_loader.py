import pandas as pd
from sentence_transformers import SentenceTransformer

# Load SBERT Model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_and_process_data(file_path="reduced_light_novels.csv"):
    """
    Loads and preprocesses the reduced light novel dataset.
    Combines the 'Genres' and 'Synopsis' fields into a single text representation
    and computes SBERT embeddings for each novel.
    """
    df = pd.read_csv(file_path)

    # Combine Genres & Synopsis for encoding (fill missing with empty string)
    df["Text_Representation"] = df["Genres"].fillna("") + " " + df["Synopsis"].fillna("")

    # Compute SBERT Embeddings (as a list of numpy arrays)
    embeddings = sbert_model.encode(df["Text_Representation"].tolist())
    
    return df, embeddings

if __name__ == "__main__":
    df, embeddings = load_and_process_data()
    print(f"Loaded {len(df)} light novels with embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
